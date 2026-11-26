# Adapted from Mohamed VehicleRec 33288b2, manufacturer/crop_quality.py; see
# docs/t1/manufacturer_quality_runtime.md. CPU-only, deterministic, no I/O.
"""Explainable manufacturer crop-quality gate.

Scores are engineering heuristics, not calibrated probabilities. A crop that
fails any hard check is ineligible and is never submitted to the model; every
rejection reason is returned so callers can retain auditable counts.
"""
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
from numbers import Real

import numpy as np

POLICY_VERSION = 'manufacturer-crop-quality-v2'
REASONS = ('empty_image', 'unsupported_array', 'too_small', 'insufficient_area',
           'implausible_car_shape', 'frame_boundary', 'severely_blurred',
           'essentially_black', 'essentially_white')


@dataclass(frozen=True)
class CropQualityConfig:
    """Versioned thresholds. Any change must change ``version``.

    v2 (default) differs from v1 only in ``severe_blur_variance`` 2.0 -> 5.0,
    calibrated on the labeled stills (docs/t1/manufacturer_quality_runtime.md).
    """
    version: str = POLICY_VERSION
    min_width: int = 192
    min_height: int = 128
    min_area: int = 98304
    min_car_aspect_ratio: float = 1.0
    car_labels: tuple = ('car',)
    max_border_edges: int = 0
    border_margin_px: int = 4
    target_width: int = 384
    target_height: int = 256
    sharpness_reference: float = 1800.0
    severe_blur_variance: float = 5.0
    exposure_low_threshold: int = 8
    exposure_high_threshold: int = 247
    exposure_poor_clipped_fraction: float = 0.50
    exposure_reject_clipped_fraction: float = 0.98
    size_weight: float = 0.30
    sharpness_weight: float = 0.35
    exposure_weight: float = 0.25
    clipping_weight: float = 0.10

    def __post_init__(self):
        if type(self.version) is not str or not self.version:
            raise ValueError('explicit quality policy version required')
        if type(self.car_labels) is not tuple or not all(type(v) is str for v in self.car_labels):
            raise ValueError('car_labels must be a tuple of class names')
        for item in fields(self):
            if item.name in ('version', 'car_labels'):
                continue
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f'{item.name} must be a finite number')
            if isinstance(item.default, int) and type(value) is not int:
                raise ValueError(f'{item.name} must be an integer')
        weights = (self.size_weight, self.sharpness_weight, self.exposure_weight, self.clipping_weight)
        if any(w < 0 for w in weights) or not np.isclose(sum(weights), 1.0):
            raise ValueError('crop quality weights must be non-negative and sum to 1')
        if min(self.min_width, self.min_height, self.target_width, self.target_height) < 1 or self.min_area < 0:
            raise ValueError('crop dimensions must be positive')
        if self.min_car_aspect_ratio < 0 or not 0 <= self.max_border_edges <= 4 or self.border_margin_px < 0:
            raise ValueError('crop shape/boundary settings are invalid')
        if self.sharpness_reference <= 0 or self.severe_blur_variance < 0:
            raise ValueError('sharpness settings are invalid')
        if not 0 <= self.exposure_low_threshold < self.exposure_high_threshold <= 255:
            raise ValueError('exposure thresholds are invalid')
        if not (0 < self.exposure_poor_clipped_fraction <= 1 and 0 < self.exposure_reject_clipped_fraction <= 1):
            raise ValueError('exposure clipped-pixel fractions are invalid')

    def value(self):
        data = asdict(self)
        data['car_labels'] = list(self.car_labels)
        return data

    def fingerprint(self):
        from contracts.db1.reference import canonical_json_bytes
        return sha256(canonical_json_bytes(self.value())).hexdigest()


@dataclass(frozen=True)
class CropQuality:
    eligible: bool
    score: float
    reasons: tuple
    width: int = 0
    height: int = 0
    sharpness_variance: float = 0.0
    dark_fraction: float = 0.0
    bright_fraction: float = 0.0
    border_edges: int | None = None
    aspect_ratio: float = 0.0

    def record(self):
        """Canonical, rounded scalar record; stable through JSON round trips."""
        return dict(eligible=self.eligible, score=round(self.score, 6), reasons=list(self.reasons),
                    width=self.width, height=self.height, sharpness_variance=round(self.sharpness_variance, 4),
                    dark_fraction=round(self.dark_fraction, 6), bright_fraction=round(self.bright_fraction, 6),
                    border_edges=self.border_edges, aspect_ratio=round(self.aspect_ratio, 6))


def _bounded(value):
    return float(np.clip(value, 0.0, 1.0))


def to_grayscale(image):
    import cv2
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0:
        return None
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0]
    else:
        return None
    return np.ascontiguousarray(gray)


def border_edges(bbox, frame_shape, margin):
    """Count frame edges touched by an already-clipped bbox (x1, y1, x2, y2)."""
    if bbox is None or frame_shape is None:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        height, width = frame_shape[:2]
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(v) for v in (x1, y1, x2, y2)) or width <= 0 or height <= 0 or x2 <= x1 or y2 <= y1:
        return None
    return int(sum((x1 <= margin, y1 <= margin, x2 >= width - margin, y2 >= height - margin)))


class CropQualityAnalyzer:
    def __init__(self, config=None):
        self.config = config or CropQualityConfig()

    def analyze(self, crop, *, bbox=None, frame_shape=None, label=None):
        import cv2
        cfg = self.config
        if isinstance(crop, np.ndarray) and crop.size == 0:
            return CropQuality(False, 0.0, ('empty_image',))
        gray = to_grayscale(crop)
        if gray is None:
            return CropQuality(False, 0.0, ('unsupported_array',))
        height, width = gray.shape
        size_score = _bounded((min(width / cfg.target_width, 1.0) + min(height / cfg.target_height, 1.0)
                               + min(width * height / (cfg.target_width * cfg.target_height), 1.0)) / 3.0)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness_score = _bounded(sharpness / (sharpness + cfg.sharpness_reference))
        dark = float(np.mean(gray <= cfg.exposure_low_threshold))
        bright = float(np.mean(gray >= cfg.exposure_high_threshold))
        exposure_score = _bounded(1.0 - (dark + bright) / cfg.exposure_poor_clipped_fraction)
        edges = border_edges(bbox, frame_shape, cfg.border_margin_px)
        clipping_score = 1.0 if edges is None else _bounded(1.0 - edges / 4.0)
        reasons = []
        if width < cfg.min_width or height < cfg.min_height:
            reasons.append('too_small')
        elif width * height < cfg.min_area:
            reasons.append('insufficient_area')
        if label in cfg.car_labels and width / height < cfg.min_car_aspect_ratio:
            reasons.append('implausible_car_shape')
        if edges is not None and edges > cfg.max_border_edges:
            reasons.append('frame_boundary')
        if sharpness <= cfg.severe_blur_variance:
            reasons.append('severely_blurred')
        if dark >= cfg.exposure_reject_clipped_fraction:
            reasons.append('essentially_black')
        elif bright >= cfg.exposure_reject_clipped_fraction:
            reasons.append('essentially_white')
        score = _bounded(cfg.size_weight * size_score + cfg.sharpness_weight * sharpness_score
                         + cfg.exposure_weight * exposure_score + cfg.clipping_weight * clipping_score)
        return CropQuality(not reasons, score, tuple(reasons), width, height, sharpness, dark, bright,
                           edges, width / height)


# Explicit, fingerprint-stable policy registry. v1 is the exact original
# configuration (fingerprint 0202982c...c7bc); it is kept for rollback/replay.
QUALITY_V1 = CropQualityConfig(version='manufacturer-crop-quality-v1', severe_blur_variance=2.0)
QUALITY_V2 = CropQualityConfig()
QUALITY_POLICIES = {'v1': QUALITY_V1, 'v2': QUALITY_V2}
