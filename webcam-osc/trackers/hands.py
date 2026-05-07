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
_DRAW_LM = _DRAW.DrawingSpec(color=(0, 255, 128), thickness=1, circle_radius=3)
_DRAW_CN = _DRAW.DrawingSpec(color=(0, 200, 80),  thickness=1)
_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS

# Landmark indices
_FINGER_MCP = {"index": 5,  "middle": 9,  "ring": 13, "pinky": 17}
_FINGER_TIP = {"index": 8,  "middle": 12, "ring": 16, "pinky": 20}
_THUMB_TIP  = 4
_INDEX_TIP  = 8
# Max normalised thumb-index distance (for pinch gesture)
_PINCH_MAX  = 0.30


def _dist3(a, b) -> float:
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)


def _angle_curl(p_tip, p_mid, p_base) -> float:
    """3-D dot-product curl at p_mid joint. 0=extended, 1=curled."""
    v1 = (p_base.x - p_mid.x, p_base.y - p_mid.y, p_base.z - p_mid.z)
    v2 = (p_tip.x  - p_mid.x, p_tip.y  - p_mid.y, p_tip.z  - p_mid.z)
    l1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2) + 1e-9
    l2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2) + 1e-9
    dot = (v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]) / (l1 * l2)
    return float(max(0.0, min(1.0, (dot + 1.0) / 2.0)))


def _finger_curl(lms, finger: str) -> float:
    """0.0 = fully extended, 1.0 = fully curled."""
    wrist = lms[0]
    if finger == "thumb":
        # Two combined signals:
        # 1. Lateral projection: project thumb tip onto the index→pinky axis
        #    (self-normalising — no magic constants, works for left/right hands).
        #    Extended outward → tip projects strongly on "index/thumb side" → curl 0
        #    Adducted/wrapped  → tip crosses toward palm centre → curl 1
        idx_mcp = lms[5]
        pky_mcp = lms[17]
        tip     = lms[4]
        ax = idx_mcp.x - pky_mcp.x
        ay = idx_mcp.y - pky_mcp.y
        az = idx_mcp.z - pky_mcp.z
        alen = math.sqrt(ax*ax + ay*ay + az*az) + 1e-9
        ax, ay, az = ax/alen, ay/alen, az/alen
        palm_cx = (idx_mcp.x + pky_mcp.x) * 0.5
        palm_cy = (idx_mcp.y + pky_mcp.y) * 0.5
        palm_cz = (idx_mcp.z + pky_mcp.z) * 0.5
        proj = ((tip.x - palm_cx)*ax + (tip.y - palm_cy)*ay + (tip.z - palm_cz)*az) / (alen * 0.5)
        lateral_curl = float(max(0.0, min(1.0, 1.0 - proj)))
        # 2. IP joint angle — catches the wrap/bend when thumb folds tightly
        ip_curl = _angle_curl(lms[4], lms[3], lms[2])
        return max(lateral_curl, ip_curl)
    else:
        d_mcp = _dist3(wrist, lms[_FINGER_MCP[finger]]) + 1e-9
        d_tip = _dist3(wrist, lms[_FINGER_TIP[finger]])
        # Extended: ratio ≈ 2.0 → curl 0; Curled: ratio ≈ 1.0 → curl 1
        return float(max(0.0, min(1.0, 2.0 - d_tip / d_mcp)))


class HandsTracker(BaseTracker):
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        gesture_threshold: float = 0.65,
    ):
        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=str(_MODELS_DIR / "hand_landmarker.task")
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._threshold = gesture_threshold
        self._max_num_hands = max_num_hands
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
        thr = self._threshold

        for i, hand_lms in enumerate(result.hand_landmarks):
            _DRAW.draw_landmarks(frame, hand_lms, _CONNECTIONS, _DRAW_LM, _DRAW_CN)

            # ── raw landmarks ────────────────────────────────────────────────
            coords = [val for lm in hand_lms for val in (lm.x, lm.y, lm.z)]
            pairs.append((f"/webcam/hand/{i}/landmarks", coords))

            # ── handedness (left=0, right=1) ─────────────────────────────────
            side = 0.0
            if result.handedness and i < len(result.handedness):
                label = result.handedness[i][0].category_name.lower()
                side = 1.0 if label == "right" else 0.0
            pairs.append((f"/webcam/hand/{i}/side", side))

            # ── fingertip positions ──────────────────────────────────────────
            for tip_id, name in ((4, "thumb"), (8, "index"), (12, "middle"), (16, "ring"), (20, "pinky")):
                lm = hand_lms[tip_id]
                pairs.append((f"/webcam/hand/{i}/tip/{name}/x", float(lm.x)))
                pairs.append((f"/webcam/hand/{i}/tip/{name}/y", float(lm.y)))

            # ── per-finger curl + open (all 5 including thumb) ───────────────
            # /curl: 0=extended  1=curled
            # /open: 0=curled    1=extended  ← map to volume/brightness
            # /open/on: 1 when finger is up
            curls = {f: _finger_curl(hand_lms, f) for f in ("thumb", "index", "middle", "ring", "pinky")}
            for fname, curl in curls.items():
                open_val = 1.0 - curl
                pairs.append((f"/webcam/hand/{i}/finger/{fname}/curl",    curl))
                pairs.append((f"/webcam/hand/{i}/finger/{fname}/open",    open_val))
                pairs.append((f"/webcam/hand/{i}/finger/{fname}/open/on", 1.0 if open_val >= thr else 0.0))

            c_idx = curls["index"]
            c_mid = curls["middle"]
            c_rng = curls["ring"]
            c_pky = curls["pinky"]

            # ── pinch ────────────────────────────────────────────────────────
            # Both conditions must hold: thumb+index close AND middle/ring/pinky open.
            # Prevents fist from triggering pinch.
            proximity   = 1.0 - min(1.0, _dist3(hand_lms[_THUMB_TIP], hand_lms[_INDEX_TIP]) / _PINCH_MAX)
            others_open = 1.0 - (c_mid + c_rng + c_pky) / 3.0
            pinch       = float(min(proximity, others_open))

            # ── composite gestures ───────────────────────────────────────────
            fist  = (c_idx + c_mid + c_rng + c_pky) / 4.0
            open_ = 1.0 - fist
            # min() ensures each condition must independently hold
            point = min(1.0 - c_idx, (c_mid + c_rng + c_pky) / 3.0)
            peace = min(1.0 - c_idx, 1.0 - c_mid, (c_rng + c_pky) / 2.0)

            for gname, gval in {"fist": fist, "open": open_, "point": point, "peace": peace, "pinch": pinch}.items():
                gval = float(max(0.0, min(1.0, gval)))
                pairs.append((f"/webcam/hand/{i}/gesture/{gname}",    gval))
                pairs.append((f"/webcam/hand/{i}/gesture/{gname}/on", 1.0 if gval >= thr else 0.0))

        # ── zero out slots for hands not detected this frame ──────────────────────
        # Ensures TouchDesigner CHOP values fall to 0 when a hand leaves frame.
        for i in range(len(result.hand_landmarks), self._max_num_hands):
            pairs.append((f"/webcam/hand/{i}/side", 0.0))
            for name in ("thumb", "index", "middle", "ring", "pinky"):
                pairs.append((f"/webcam/hand/{i}/tip/{name}/x",          0.0))
                pairs.append((f"/webcam/hand/{i}/tip/{name}/y",          0.0))
                pairs.append((f"/webcam/hand/{i}/finger/{name}/curl",    0.0))
                pairs.append((f"/webcam/hand/{i}/finger/{name}/open",    0.0))
                pairs.append((f"/webcam/hand/{i}/finger/{name}/open/on", 0.0))
            for gname in ("fist", "open", "point", "peace", "pinch"):
                pairs.append((f"/webcam/hand/{i}/gesture/{gname}",    0.0))
                pairs.append((f"/webcam/hand/{i}/gesture/{gname}/on", 0.0))

        return pairs

    def release(self):
        self._landmarker.close()
