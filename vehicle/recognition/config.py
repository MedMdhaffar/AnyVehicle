"""Explicit artifact locations shared by the engine and launcher."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def boolean(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in ('1', 'true'):
        return True
    if str(value).lower() in ('0', 'false'):
        return False
    raise ValueError('Expected true/false or 1/0')


def paths():
    vehicle = Path(__file__).resolve().parent/'attributes'
    plate = Path(__file__).resolve().parent/'plate'
    defaults = dict(vehicle_model=vehicle/'models/vehicle_yolo26.pt',
                    vehicle_tracker=vehicle/'trackers/bytetrack_vehicles.yaml',
                    plate_model=plate/'models/plate_detector.onnx',
                    plate_tracker=plate/'trackers/bytetrack_plates.yaml',
                    marker_model=plate/'models/tounes_word_detector.pt',
                    ocr_dir=plate/'models/PP-OCRv6_medium_rec')
    result = {}
    for key, path in defaults.items():
        path = Path(os.environ.get('T1_'+key.upper(),path)).expanduser()
        result[key] = (path if path.is_absolute() else ROOT/path).resolve()
    return result


def plate_config():
    from types import SimpleNamespace
    from .plate import config
    values = {k: getattr(config, k) for k in dir(config) if k.isupper()}
    p = paths()
    values.update(PLATE_DETECTOR_PATH=str(p['plate_model']), WORD_DETECTOR_PATH=str(p['marker_model']),
                  REGISTRATION_OCR_MODEL_DIR=p['ocr_dir'],
                  TUNIS_USE_LEGACY_WHOLE_OCR=boolean(os.environ.get('TUNIS_USE_LEGACY_WHOLE_OCR', 'false')))
    return SimpleNamespace(**values)
