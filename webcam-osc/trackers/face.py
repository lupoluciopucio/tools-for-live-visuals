import math
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
_DRAW_LM = _DRAW.DrawingSpec(color=(0, 200, 255), thickness=1, circle_radius=1)
_DRAW_CN = _DRAW.DrawingSpec(color=(0, 150, 200), thickness=1)
_CONNECTIONS = vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS

# Fallback landmark indices if blendshapes unavailable
_UPPER_LIP = 13
_LOWER_LIP = 14
_LEFT_EYE_TOP = 159
_LEFT_BROW_TOP = 105
_FACE_HEIGHT_TOP = 10
_FACE_HEIGHT_BOT = 152


class FaceTracker(BaseTracker):
    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        send_raw_landmarks: bool = True,
        landmark_chunk_size: int = 200,
    ):
        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=str(_MODELS_DIR / "face_landmarker.task")
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._send_raw = send_raw_landmarks
        self._chunk_size = landmark_chunk_size
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

        if not result.face_landmarks:
            return pairs

        face_lms = result.face_landmarks[0]
        _DRAW.draw_landmarks(frame, face_lms, _CONNECTIONS, _DRAW_LM, _DRAW_CN)

        # Derived values: prefer blendshapes (ARKit-accurate), fall back to geometry
        if result.face_blendshapes and result.face_blendshapes[0]:
            bs = {b.category_name: b.score for b in result.face_blendshapes[0]}
            mouth_open = float(bs.get("jawOpen", 0.0))
            brow_raise = float(max(
                bs.get("browOuterUpLeft",  0.0),
                bs.get("browOuterUpRight", 0.0),
            ))
        else:
            lm = face_lms
            face_h = abs(lm[_FACE_HEIGHT_BOT].y - lm[_FACE_HEIGHT_TOP].y) or 1.0
            mouth_open = abs(lm[_LOWER_LIP].y - lm[_UPPER_LIP].y) / face_h
            brow_raise  = abs(lm[_LEFT_BROW_TOP].y - lm[_LEFT_EYE_TOP].y) / face_h

        pairs.append(("/webcam/face/mouth_open", mouth_open))
        pairs.append(("/webcam/face/brow_raise", brow_raise))

        nose = face_lms[1]
        chin = face_lms[152]
        tilt_rad = math.atan2(chin.x - nose.x, chin.y - nose.y)
        pairs.append(("/webcam/face/head_tilt", float(math.degrees(tilt_rad))))

        if self._send_raw:
            coords = [val for lm in face_lms for val in (lm.x, lm.y, lm.z)]
            chunk = self._chunk_size
            for i in range(0, len(coords), chunk):
                pairs.append((f"/webcam/face/landmarks/{i // chunk}", coords[i : i + chunk]))

        return pairs

    def release(self):
        self._landmarker.close()
