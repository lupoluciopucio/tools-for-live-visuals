import time
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import numpy as np

from trackers.base import BaseTracker

_MODELS_DIR = Path(__file__).parent.parent / "models"
_DRAW = vision.drawing_utils
_DRAW_LM = _DRAW.DrawingSpec(color=(255, 100, 0), thickness=2, circle_radius=3)
_DRAW_CN = _DRAW.DrawingSpec(color=(255, 200, 0), thickness=2)
_CONNECTIONS = vision.PoseLandmarksConnections.POSE_LANDMARKS

_KEY_JOINTS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}


class PoseTracker(BaseTracker):
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ):
        # model_complexity 0=lite, 1=full (2=heavy not downloaded by default)
        model_file = "pose_landmarker_full.task"
        options = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=str(_MODELS_DIR / model_file)
            ),
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._t0 = time.perf_counter()
        self._last_ts: int = -1

    def _ts(self) -> int:
        ts = int((time.perf_counter() - self._t0) * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        return ts

    def process(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray | None = None,
    ) -> list[tuple[str, Any]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, self._ts())

        pairs: list[tuple[str, Any]] = []

        if not result.pose_landmarks:
            return pairs

        pose_lms = result.pose_landmarks[0]
        _DRAW.draw_landmarks(frame, pose_lms, _CONNECTIONS, _DRAW_LM, _DRAW_CN)

        coords = [val for lm in pose_lms for val in (lm.x, lm.y, lm.z, lm.visibility)]
        pairs.append(("/webcam/pose/landmarks", coords))

        for name, idx in _KEY_JOINTS.items():
            lm = pose_lms[idx]
            pairs.append((f"/webcam/pose/{name}/x", float(lm.x)))
            pairs.append((f"/webcam/pose/{name}/y", float(lm.y)))

        return pairs

    def release(self):
        self._landmarker.close()
