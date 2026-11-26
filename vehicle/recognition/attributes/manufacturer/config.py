"""Bounded local inference configuration."""
import json
import math
import os
from dataclasses import dataclass, fields

from pathlib import Path
from .gguf_assets import MODEL_FILENAME, PROJECTOR_FILENAME
from .crop_quality import CropQualityConfig

MODEL_ID = "LiquidAI/LFM2.5-VL-450M"

MODEL_PARAMETERS = (
    {"key": "temperature", "label": "Temperature", "min": 0, "max": 2, "step": 0.1,
     "description": "0 selects the most likely answer. Higher values add variation."},
    {"key": "top_p", "label": "Top-p", "min": 0.05, "max": 1, "step": 0.05,
     "description": "Limits token choices when temperature is above 0."},
    {"key": "max_new_tokens", "label": "Output token limit", "min": 1, "max": 128, "step": 1,
     "description": "Maximum generated tokens per recognition attempt."},
    {"key": "max_attempts", "label": "Attempts per vehicle", "min": 1, "max": 5, "step": 1,
     "description": "Maximum recognition calls for each track, including retries."},
)

QUALITY_LIMITS = {
    "min_width": (1, 8192), "min_height": (1, 8192),
    "min_area": (0, 8192 * 8192),
    "min_car_aspect_ratio": (0, 3), "max_border_edges": (0, 4),
    "target_width": (1, 8192), "target_height": (1, 8192),
    "sharpness_reference": (1e-9, 1e9), "severe_blur_variance": (0, 1e9),
    "exposure_low_threshold": (0, 255), "exposure_high_threshold": (0, 255),
    "exposure_poor_clipped_fraction": (1e-9, 1),
    "exposure_reject_clipped_fraction": (1e-9, 1),
    "border_margin_px": (0, 8192),
    "size_weight": (0, 1), "sharpness_weight": (0, 1),
    "exposure_weight": (0, 1), "clipping_weight": (0, 1),
}
MODEL_PARAMETERS += tuple(
    {"key": "quality_" + item.name, "label": "Crop quality: " + item.name.replace("_", " "),
     "min": QUALITY_LIMITS[item.name][0], "max": QUALITY_LIMITS[item.name][1],
     "step": 1 if isinstance(item.default, int) else 0.01,
     "description": "Uncalibrated crop-quality baseline; measured before VLM resizing."}
    for item in fields(CropQualityConfig)
)


def parse_model_settings(raw):
    if raw is None:
        return {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError as exc:
            raise ValueError("model_settings must be a JSON object") from exc
    if not isinstance(raw, dict):
        raise ValueError("model_settings must be a JSON object")
    parameters = {field["key"]: field for field in MODEL_PARAMETERS}
    for key, value in raw.items():
        if key not in parameters:
            raise ValueError(f"Unknown model parameter: {key}")
        field = parameters[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not field["min"] <= value <= field["max"] or not math.isfinite(value)
                or (field["step"] == 1 and not isinstance(value, int))):
            raise ValueError(f"{key} must be {'an integer' if field['step'] == 1 else 'a number'} between {field['min']} and {field['max']}")
    quality = {item.name: raw.get("quality_" + item.name, item.default)
               for item in fields(CropQualityConfig)}
    CropQualityConfig(**quality)
    return dict(raw)


@dataclass(frozen=True)
class RecognitionConfig:
    # Select "huggingface" (Transformers) or "llama.cpp".
    engine: str = "llama.cpp"
    hf_model_path: str = str(Path(__file__).resolve().parents[1] / "models" / "LFM2.5-VL-450M")
    model_path: str = ""
    projector_path: str = ""
    context_size: int = 4096
    device: str = "cuda:0"
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 128
    max_attempts: int = 3
    crop_gap: int = 8
    initial_frames: int = 8
    max_crop_side: int = 0

    max_pending_crop_mb: int = 256
    quality_min_width: int = CropQualityConfig.min_width
    quality_min_height: int = CropQualityConfig.min_height
    quality_min_area: int = CropQualityConfig.min_area
    quality_min_car_aspect_ratio: float = CropQualityConfig.min_car_aspect_ratio
    quality_max_border_edges: int = CropQualityConfig.max_border_edges
    quality_target_width: int = CropQualityConfig.target_width
    quality_target_height: int = CropQualityConfig.target_height
    quality_sharpness_reference: float = 1800.0
    quality_severe_blur_variance: float = 2.0
    quality_exposure_low_threshold: int = 8
    quality_exposure_high_threshold: int = 247
    quality_exposure_poor_clipped_fraction: float = 0.50
    quality_exposure_reject_clipped_fraction: float = 0.98
    quality_border_margin_px: int = 4
    quality_size_weight: float = 0.30
    quality_sharpness_weight: float = 0.35
    quality_exposure_weight: float = 0.25
    quality_clipping_weight: float = 0.10

    def __post_init__(self):
        if self.engine not in {"huggingface", "llama.cpp"}:
            raise ValueError('engine must be "huggingface" or "llama.cpp"')
        parse_model_settings(self.model_settings())
        if isinstance(self.context_size, bool) or not isinstance(self.context_size, int) or self.context_size < 512:
            raise ValueError("context_size must be an integer of at least 512 tokens")
        for name, maximum in (("max_new_tokens", 128), ("max_attempts", 5),
                              ("crop_gap", 300), ("initial_frames", 300),
                              ("max_pending_crop_mb", 1024)):
            if not 1 <= getattr(self, name) <= maximum:
                raise ValueError(f"{name} must be between 1 and {maximum}")
        if self.max_crop_side != 0 and not 48 <= self.max_crop_side <= 8192:
            raise ValueError("max_crop_side must be 0 (original resolution) or between 48 and 8192")

    def model_settings(self):
        return {field["key"]: getattr(self, field["key"]) for field in MODEL_PARAMETERS}

    def quality_config(self):
        return CropQualityConfig(**{item.name: getattr(self, "quality_" + item.name)
                                    for item in fields(CropQualityConfig)})

    @classmethod
    def from_env(cls):
        directory = Path(__file__).resolve().parents[1] / "models"
        return cls(engine=os.getenv("MANUFACTURER_ENGINE", cls.engine),
                   hf_model_path=os.getenv("MANUFACTURER_HF_MODEL_PATH", cls.hf_model_path),
                   model_path=os.getenv("MANUFACTURER_MODEL_PATH", str(directory / MODEL_FILENAME)),
                   projector_path=os.getenv("MANUFACTURER_PROJECTOR_PATH", str(directory / PROJECTOR_FILENAME)),
                   context_size=int(os.getenv("MANUFACTURER_CONTEXT_SIZE", "4096")),
                   device=os.getenv("MANUFACTURER_DEVICE", "cuda:0"),
                   temperature=float(os.getenv("MANUFACTURER_TEMPERATURE", "0")),
                   top_p=float(os.getenv("MANUFACTURER_TOP_P", "1")),
                   **{name: int(os.getenv("MANUFACTURER_" + name.upper(), str(default)))
                      for name, default in (("max_new_tokens", 128), ("max_attempts", 3),
                          ("crop_gap", 8), ("initial_frames", 8), ("max_crop_side", 0), ("max_pending_crop_mb", 256))},
                   **{"quality_" + item.name: (int if isinstance(item.default, int) else float)(
                       os.getenv("MANUFACTURER_QUALITY_" + item.name.upper(), str(item.default)))
                      for item in fields(CropQualityConfig)})

def recognition_enabled():
    value = os.getenv("MANUFACTURER_ENABLED", "1").lower()
    if value not in {"0", "1", "true", "false"}:
        raise ValueError("MANUFACTURER_ENABLED must be 0, 1, true or false")
    return value in {"1", "true"}
