"""Deterministic, explainable crop-quality analysis for manufacturer evidence."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import cv2
import numpy as np

BBox = tuple[float, float, float, float]
FrameSize = tuple[int, int]  # width, height


@dataclass(frozen=True, eq=False)
class CropCandidate:
    """A decoded vehicle crop plus optional source-frame metadata.

    The image is the only required value and is expected to be an OpenCV-style uint8
    array. No ByteTrack or provider-specific object is required.
    """

    image: np.ndarray
    frame_index: int | None = None
    candidate_id: str | int | None = None
    bbox: BBox | None = None
    frame_size: FrameSize | None = None
    source_quality: CropQuality | None = None
    detector_label: str | None = None
    detector_confidence: float | None = None


@dataclass(frozen=True)
class CropQuality:
    eligible: bool
    score: float
    size_score: float
    sharpness_score: float
    exposure_score: float
    clipping_score: float
    reasons: tuple[str, ...] = ()
    original_width: int = 0
    original_height: int = 0
    original_area: int = 0
    sharpness_variance: float = 0.0
    dark_fraction: float = 0.0
    bright_fraction: float = 0.0
    boundary_contact: bool | None = None
    border_edges: int | None = None
    aspect_ratio: float = 0.0


@dataclass(frozen=True)
class CropQualityConfig:
    """Conservative engineering defaults; scores are not calibrated probabilities."""

    # Recognition needs more source detail than the previous 48x32 validity floor.
    # These are conservative starting values, not calibrated recognizability gates.
    min_width: int = 192
    min_height: int = 128
    min_area: int = 98_304
    min_car_aspect_ratio: float = 1.0
    max_border_edges: int = 0
    target_width: int = 384
    target_height: int = 256
    sharpness_reference: float = 1800.0
    severe_blur_variance: float = 2.0
    exposure_low_threshold: int = 8
    exposure_high_threshold: int = 247
    exposure_poor_clipped_fraction: float = 0.50
    exposure_reject_clipped_fraction: float = 0.98
    border_margin_px: int = 4
    size_weight: float = 0.30
    sharpness_weight: float = 0.35
    exposure_weight: float = 0.25
    clipping_weight: float = 0.10

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            default = self.__dataclass_fields__[name].default
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if isinstance(default, int) and not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        weights = (
            self.size_weight,
            self.sharpness_weight,
            self.exposure_weight,
            self.clipping_weight,
        )
        if any(weight < 0 for weight in weights) or not np.isclose(sum(weights), 1.0):
            raise ValueError("crop quality weights must be non-negative and sum to 1")
        if min(self.min_width, self.min_height, self.target_width, self.target_height) < 1:
            raise ValueError("crop dimensions must be positive")
        if self.min_area < 0:
            raise ValueError("minimum crop area must be non-negative")
        if self.min_car_aspect_ratio < 0 or not 0 <= self.max_border_edges <= 4:
            raise ValueError("crop shape/boundary settings are invalid")
        if self.sharpness_reference <= 0 or self.severe_blur_variance < 0:
            raise ValueError("sharpness settings are invalid")
        if not (
            0
            <= self.exposure_low_threshold
            < self.exposure_high_threshold
            <= 255
        ):
            raise ValueError("exposure thresholds are invalid")
        if not (
            0 < self.exposure_poor_clipped_fraction <= 1
            and 0 < self.exposure_reject_clipped_fraction <= 1
        ):
            raise ValueError("exposure clipped-pixel fractions are invalid")
        if self.border_margin_px < 0:
            raise ValueError("border margin must be non-negative")


def _bounded(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _neutral_failure(*reasons: str) -> CropQuality:
    return CropQuality(
        eligible=False,
        score=0.0,
        size_score=0.0,
        sharpness_score=0.0,
        exposure_score=0.0,
        clipping_score=1.0,
        reasons=tuple(reasons),
    )


def to_grayscale(image: object) -> np.ndarray | None:
    """Normalize supported uint8 gray/BGR/BGRA arrays without changing the input."""

    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0:
        return None
    try:
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3 and image.shape[2] == 1:
            gray = image[:, :, 0]
        elif image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 3 and image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            return None
    except (cv2.error, ValueError):
        return None
    if gray.ndim != 2 or gray.shape[0] == 0 or gray.shape[1] == 0:
        return None
    return np.ascontiguousarray(gray)


class CropQualityAnalyzer:
    """Scores one crop using size, sharpness, exposure, and clipping signals."""

    def __init__(self, config: CropQualityConfig | None = None) -> None:
        self.config = config or CropQualityConfig()

    def analyze(self, candidate: CropCandidate) -> CropQuality:
        image = candidate.image
        if isinstance(image, np.ndarray) and image.size == 0:
            return _neutral_failure("empty_image")
        gray = to_grayscale(image)
        if gray is None:
            return _neutral_failure("unsupported_array")

        height, width = gray.shape
        cfg = self.config
        size_score = self._size_score(width, height)
        sharpness_raw = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness_score = _bounded(
            sharpness_raw / (sharpness_raw + cfg.sharpness_reference)
        )
        dark_fraction, bright_fraction = self._clipped_fractions(gray)
        exposure_score = self._exposure_score(dark_fraction, bright_fraction)
        border_edges = self._border_contacts(candidate)
        clipping_score = 1.0 if border_edges is None else _bounded(1.0 - border_edges / 4.0)

        reasons: list[str] = []
        if width < cfg.min_width or height < cfg.min_height:
            reasons.append("too_small")
        elif width * height < cfg.min_area:
            reasons.append("insufficient_area")
        if candidate.detector_label == "car" and width / height < cfg.min_car_aspect_ratio:
            reasons.append("implausible_car_shape")
        if border_edges is not None and border_edges > cfg.max_border_edges:
            reasons.append("frame_boundary")
        if sharpness_raw <= cfg.severe_blur_variance:
            reasons.append("severely_blurred")
        if dark_fraction >= cfg.exposure_reject_clipped_fraction:
            reasons.append("essentially_black")
        elif bright_fraction >= cfg.exposure_reject_clipped_fraction:
            reasons.append("essentially_white")

        score = _bounded(
            cfg.size_weight * size_score
            + cfg.sharpness_weight * sharpness_score
            + cfg.exposure_weight * exposure_score
            + cfg.clipping_weight * clipping_score
        )
        return CropQuality(
            eligible=not reasons,
            score=score,
            size_score=size_score,
            sharpness_score=sharpness_score,
            exposure_score=exposure_score,
            clipping_score=clipping_score,
            reasons=tuple(reasons),
            original_width=width,
            original_height=height,
            original_area=width * height,
            sharpness_variance=sharpness_raw,
            dark_fraction=dark_fraction,
            bright_fraction=bright_fraction,
            boundary_contact=border_edges > 0 if border_edges is not None else None,
            border_edges=border_edges,
            aspect_ratio=width / height,
        )

    def _size_score(self, width: int, height: int) -> float:
        cfg = self.config
        width_score = min(width / cfg.target_width, 1.0)
        height_score = min(height / cfg.target_height, 1.0)
        area_score = min(
            (width * height) / (cfg.target_width * cfg.target_height), 1.0
        )
        return _bounded((width_score + height_score + area_score) / 3.0)

    def _clipped_fractions(self, gray: np.ndarray) -> tuple[float, float]:
        cfg = self.config
        dark_fraction = float(np.mean(gray <= cfg.exposure_low_threshold))
        bright_fraction = float(np.mean(gray >= cfg.exposure_high_threshold))
        return dark_fraction, bright_fraction

    def _exposure_score(self, dark_fraction: float, bright_fraction: float) -> float:
        clipped_fraction = dark_fraction + bright_fraction
        return _bounded(
            1.0 - clipped_fraction / self.config.exposure_poor_clipped_fraction
        )

    def _border_contacts(self, candidate: CropCandidate) -> int | None:
        if candidate.bbox is None or candidate.frame_size is None:
            return None
        try:
            if len(candidate.bbox) != 4 or len(candidate.frame_size) != 2:
                return None
            values = (*candidate.bbox, *candidate.frame_size)
            if any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(float(value))
                for value in values
            ):
                return None
            x1, y1, x2, y2, frame_width, frame_height = map(float, values)
            if (
                frame_width <= 0
                or frame_height <= 0
                or x2 <= x1
                or y2 <= y1
            ):
                return None

            # Detector boxes may extend beyond the source frame. Analyze the visible
            # intersection; a fully non-intersecting box provides no usable clipping
            # metadata and therefore remains neutral.
            x1 = min(max(x1, 0.0), frame_width)
            y1 = min(max(y1, 0.0), frame_height)
            x2 = min(max(x2, 0.0), frame_width)
            y2 = min(max(y2, 0.0), frame_height)
            if x2 <= x1 or y2 <= y1:
                return None
        except (TypeError, ValueError, OverflowError):
            return None

        margin = self.config.border_margin_px
        touched_edges = sum(
            (
                x1 <= margin,
                y1 <= margin,
                x2 >= frame_width - margin,
                y2 >= frame_height - margin,
            )
        )
        return touched_edges
