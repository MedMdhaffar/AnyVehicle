"""Synchronous bounded crop collection and independent evidence consensus."""
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import json
import time
import logging
from pathlib import Path
import cv2
from .config import MODEL_ID
from .crop_quality import CropCandidate
from .crop_selector import BestCropSelector, BestCropSelectorConfig, visual_signature, signature_distance

log = logging.getLogger(__name__)

@dataclass
class TrackRecognition:
    status: str = "disabled"
    manufacturer: str = "unknown"
    body_type: str = "unknown"
    color: str = "unknown"
    attempts: list = field(default_factory=list)
    candidate: object = None
    quality: float = -1.0
    last_frame: int = -100000
    used_signatures: list = field(default_factory=list)
    last_quality: float = -1.0
    failure_reason: str = ""
    key: str = ""
    crops: dict = field(default_factory=dict)
    best_saved_quality: float = -1.0
    track_id: int | None = None
    observed_crops: int = 0
    accepted_crops: int = 0
    rejected_crops: int = 0
    rejection_reasons: dict = field(default_factory=dict)
    scheduling_reasons: dict = field(default_factory=dict)
    quality_seconds: float = 0.0
    candidate_decision: dict | None = None
    selected_frame: int | None = None

    def to_dict(self):
        successes = [a for a in self.attempts if "result" in a]
        return {"manufacturer": self.manufacturer, "vlm_vehicle_type": self.body_type, "color": self.color,
                "recognition_status": self.status,
                "decision_source": "local_lfm" if successes else "none",
                "recognition_attempts": len(self.attempts), "supporting_crops": len(successes),
                "manufacturer_supporting_crops": sum(a["result"]["manufacturer"] == self.manufacturer for a in successes) if self.manufacturer != "unknown" else 0,
                "body_type_supporting_crops": sum(a["result"]["vehicle_type"] == self.body_type for a in successes) if self.body_type != "unknown" else 0,
                "color_supporting_crops": sum(a["result"]["color"] == self.color for a in successes) if self.color != "unknown" else 0,
                "recognition_seconds": sum(a["seconds"] for a in self.attempts),
                "recognition_model": MODEL_ID if self.status != "disabled" else None,
                "selected_crops": self.crops, "recognition_evidence": self.attempts, "recognition_failure": self.failure_reason,
                "crop_quality": {"observed": self.observed_crops, "accepted": self.accepted_crops,
                                 "rejected": self.rejected_crops, "rejection_reasons": self.rejection_reasons,
                                 "scheduling_reasons": self.scheduling_reasons, "seconds": self.quality_seconds,
                                 "selected_frame": self.selected_frame}}

class TrackRecognizer:
    def __init__(self, recognizer, config):
        self.recognizer, self.config = recognizer, config
        self.selector = BestCropSelector(BestCropSelectorConfig(quality=config.quality_config()))
        self.output_dir = None
        self.frame_rate = None
        self.decision_sink = None

    def emit_decision(self, record, image=None):
        if self.output_dir is not None:
            path = Path(self.output_dir).parent / "quality_decisions.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        if self.decision_sink is not None:
            self.decision_sink(record, image)

    def decision_record(self, state, bbox, frame_index):
        return {"event": "crop_decision", "candidate_id": f"{state.key}:{frame_index}",
                "track_id": state.track_id, "track_key": state.key, "frame_index": frame_index,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "video_timestamp_seconds": frame_index / self.frame_rate if self.frame_rate else None,
                "bbox": list(bbox), "retained": False, "submitted": False}

    def resource_skip(self, state, bbox, frame_index, frame_size=None):
        state.observed_crops += 1
        state.failure_reason = "crop_memory_budget_exceeded"
        reason = state.failure_reason
        state.scheduling_reasons[reason] = state.scheduling_reasons.get(reason, 0) + 1
        record = {**self.decision_record(state, bbox, frame_index), "frame_size": frame_size,
                  "eligible": None, "decision_reason": reason,
                  "sharpness_variance": None, "dark_fraction": None, "bright_fraction": None}
        if frame_size is not None:
            width, height = frame_size
            x1, y1, x2, y2 = bbox
            crop_width = max(0, min(width, x2) - max(0, x1))
            crop_height = max(0, min(height, y2) - max(0, y1))
            record.update(original_width=crop_width, original_height=crop_height,
                          original_area=crop_width * crop_height)
        self.emit_decision(record)

    def observe(self, state, frame, bbox, frame_index, frames_seen, *, detector_label=None, detector_confidence=None):
        state.observed_crops += 1
        record = self.decision_record(state, bbox, frame_index)
        record.update(detector_label=detector_label, detector_confidence=detector_confidence)
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            state.scheduling_reasons["invalid_crop"] = state.scheduling_reasons.get("invalid_crop", 0) + 1
            self.emit_decision({**record, "eligible": None, "decision_reason": "invalid_crop"})
            return
        crop = frame[y1:y2, x1:x2]
        source = CropCandidate(crop, frame_index, candidate_id=record["candidate_id"],
                               bbox=tuple(bbox), frame_size=(width, height),
                               detector_label=detector_label, detector_confidence=detector_confidence)
        started = time.perf_counter()
        selection = self.selector.select([source], max_crops=1, min_crops=1)
        quality = selection.evaluations[0].quality
        elapsed = time.perf_counter() - started
        state.quality_seconds += elapsed
        record.update(asdict(quality), frame_size=[width, height], quality_seconds=elapsed,
                      vlm_width=None, vlm_height=None)
        if not quality.eligible:
            state.rejected_crops += 1
            for reason in quality.reasons:
                state.rejection_reasons[reason] = state.rejection_reasons.get(reason, 0) + 1
            self.emit_decision({**record, "decision_reason": "quality_rejected"}, crop)
            return
        state.accepted_crops += 1
        if state.failure_reason in {"insufficient_quality", "no_usable_crop"}:
            state.failure_reason = ""
        reason = None
        if len(state.attempts) >= self.config.max_attempts:
            reason = "attempt_budget_exhausted"
        elif frame_index - state.last_frame < self.config.crop_gap:
            reason = "cooldown"
        elif crop.nbytes > self.config.max_pending_crop_mb * 1024 * 1024:
            reason = "crop_memory_budget_exceeded"
            state.failure_reason = reason
        if reason:
            state.scheduling_reasons[reason] = state.scheduling_reasons.get(reason, 0) + 1
            self.emit_decision({**record, "decision_reason": reason}, crop)
            return
        state.failure_reason = ""
        state.last_frame = frame_index
        scale = min(1.0, self.config.max_crop_side / max(crop.shape[:2])) if self.config.max_crop_side else 1.0
        crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))) if scale < 1 else crop.copy()
        candidate = replace(source, image=crop, source_quality=quality)
        record.update(vlm_width=crop.shape[1], vlm_height=crop.shape[0])
        score = quality.score
        if "early" not in state.crops:
            self.save_crop(state, "early", crop)
        if score > state.best_saved_quality:
            self.save_crop(state, "best", crop)
            state.best_saved_quality = score
        signature = visual_signature(crop)
        if any(signature_distance(signature, previous) <= 0.20 for previous in state.used_signatures):
            state.scheduling_reasons["duplicate"] = state.scheduling_reasons.get("duplicate", 0) + 1
            self.emit_decision({**record, "decision_reason": "duplicate"}, crop)
            return
        if score > state.quality:
            state.candidate, state.quality = candidate, score
            record["retained"] = True
            state.candidate_decision = dict(record)
        record["decision_reason"] = "retained" if record["retained"] else "lower_rank"
        self.emit_decision(record, crop)
        initial = not state.attempts and frames_seen >= self.config.initial_frames
        better = bool(state.attempts) and state.quality > state.last_quality + 0.05
        retry = bool(state.attempts) and state.attempts[-1].get("error") == "invalid_model_json"
        if initial or better or retry:
            self.attempt(state)

    def save_crop(self, state, label, image):
        if self.output_dir is None:
            return None
        directory = Path(self.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"{state.key}_{label}.jpg"
        if not cv2.imwrite(str(directory / filename), image):
            raise RuntimeError("Could not save vehicle crop")
        state.crops[label] = filename
        return filename

    def attempt(self, state):
        if state.candidate is None or len(state.attempts) >= self.config.max_attempts:
            return
        candidate = state.candidate
        if candidate.source_quality is None or not candidate.source_quality.eligible:
            raise ValueError("Pending vehicle crop must have eligible source quality")
        state.candidate = None
        state.last_quality = state.quality
        state.quality = -1.0
        started = time.perf_counter()
        state.selected_frame = candidate.frame_index
        submitted = {**state.candidate_decision, "event": "crop_submitted", "submitted": True}
        record = {"frame_index": candidate.frame_index, "quality_score": state.last_quality,
                  "source_quality": asdict(candidate.source_quality),
                  "detector_label": candidate.detector_label, "detector_confidence": candidate.detector_confidence,
                  "vlm_width": candidate.image.shape[1], "vlm_height": candidate.image.shape[0]}
        state.candidate_decision = None
        log.info("%s: VLM attempt %d/%d frame=%s crop=%dx%d quality=%.3f", state.key, len(state.attempts) + 1, self.config.max_attempts, candidate.frame_index, candidate.image.shape[1], candidate.image.shape[0], state.last_quality)
        try:
            record["crop"] = self.save_crop(state, f"attempt_{len(state.attempts) + 1}", candidate.image)
            self.emit_decision(submitted, candidate.image)
            retry_invalid = bool(state.attempts) and state.attempts[-1].get("error") == "invalid_model_json"
            if retry_invalid:
                record["retry_reason"] = "invalid_model_json"
                result = self.recognizer.predict(candidate.image, retry_invalid=True)
            else:
                result = self.recognizer.predict(candidate.image)
            record["result"] = result.to_dict()
            state.used_signatures.append(visual_signature(candidate.image))
        except Exception as exc:
            record["error"] = str(exc) if isinstance(exc, ValueError) and str(exc) in {"invalid_model_json", "invalid_crop"} else type(exc).__name__
            if hasattr(exc, "raw_output"):
                record["raw_output"] = exc.raw_output
            log.warning("%s: recognition attempt failed (%s); raw_output=%r", state.key, record["error"], record.get("raw_output", "unavailable"))
        record["seconds"] = time.perf_counter() - started
        state.attempts.append(record)
        self.aggregate(state)
        log.info("%s: attempt finished in %.2fs; prediction=%s selected=%s/%s/%s status=%s", state.key, record["seconds"], record.get("result", record.get("error")), state.manufacturer, state.body_type, state.color, state.status)

    @staticmethod
    def aggregate(state):
        valid = [a["result"] for a in state.attempts if "result" in a]
        selected = valid[-1] if valid else {}
        for key, attr in (("manufacturer", "manufacturer"), ("vehicle_type", "body_type"), ("color", "color")):
            setattr(state, attr, selected.get(key, "unknown"))
        state.status = "recognized" if state.manufacturer != "unknown" or state.body_type != "unknown" or state.color != "unknown" else ("unknown" if valid else "error")

    def finish(self, state):
        self.attempt(state)
        state.candidate = None
        if not state.attempts:
            state.status = "unknown"
            state.failure_reason = state.failure_reason or (
                "insufficient_quality" if state.rejected_crops and not state.accepted_crops else "no_usable_crop")
        elif state.status == "unknown":
            state.failure_reason = "no_known_vlm_labels"
        elif state.status == "error":
            state.failure_reason = state.attempts[-1].get("error", "inference_failure")
        self.emit_decision({"event": "track_final", "track_id": state.track_id,
                            "track_key": state.key, **state.to_dict()})
