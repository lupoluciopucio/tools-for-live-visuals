"""Downloads mediapipe .task model files on first run."""
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"

_URLS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    ),
    "pose_landmarker_full.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    ),
}

_TRACKER_MODEL = {
    "hands": "hand_landmarker.task",
    "face":  "face_landmarker.task",
    "pose":  "pose_landmarker_full.task",
}


def ensure_models(tracker_cfg: dict):
    """Download model files for each enabled tracker that isn't already cached."""
    MODELS_DIR.mkdir(exist_ok=True)
    for tracker_key, model_name in _TRACKER_MODEL.items():
        if not tracker_cfg.get(tracker_key, {}).get("enabled", False):
            continue
        path = MODELS_DIR / model_name
        if path.exists():
            continue
        url = _URLS[model_name]
        size_mb = {"hand_landmarker.task": 9, "face_landmarker.task": 30, "pose_landmarker_full.task": 29}
        print(f"  Downloading {model_name} (~{size_mb.get(model_name, '?')} MB)…", flush=True)

        def _progress(count, block_size, total_size):
            if total_size > 0:
                pct = min(100, count * block_size * 100 // total_size)
                print(f"\r    {pct}%", end="", flush=True)

        urllib.request.urlretrieve(url, path, reporthook=_progress)
        print(f"\r  Saved: {path}")
