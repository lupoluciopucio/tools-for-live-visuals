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


def _finger_curl(lms, finger: str) -> float:
    """0.0 = fully extended, 1.0 = fully curled."""
    wrist = lms[0]
    if finger == "thumb":
        palm_len = _dist3(wrist, lms[9]) + 1e-9
        tt = lms[4]
        # Min distance from thumb tip to ALL 8 non-thumb anchor landmarks
        # (4 MCPs + 4 PIPs).
        # - MCPs are fixed base knuckles → catch "open hand, thumb adducted"
        # - PIPs fold inward in a fist → catch "fist, thumb wrapping over fingers"
        # Together they cover the entire non-thumb palm area, so the thumb tip
        # is near at least one of them in ANY closed position, and far from all
        # of them only when truly extended/abducted.
        d_min = min(
            _dist3(tt, lms[5]),   _dist3(tt, lms[9]),   # index / middle MCP
            _dist3(tt, lms[13]),  _dist3(tt, lms[17]),  # ring   / pinky  MCP
            _dist3(tt, lms[6]),   _dist3(tt, lms[10]),  # index / middle  PIP
            _dist3(tt, lms[14]),  _dist3(tt, lms[18]),  # ring  / pinky   PIP
        ) / palm_len
        open_lin = float(max(0.0, min(1.0, (d_min - 0.10) / 0.35)))
        return 1.0 - open_lin ** 1.5
    else:
        d_mcp = _dist3(wrist, lms[_FINGER_MCP[finger]]) + 1e-9
        d_tip = _dist3(wrist, lms[_FINGER_TIP[finger]])
        ratio = d_tip / d_mcp
        # Map [1.0=curled, 1.8=extended] → curl [1.0, 0.0].
        # Using 1.8 instead of 2.0 so that the pinky (whose tip-to-wrist only
        # reaches ≈1.7× MCP distance when fully extended) still reads 0 curl.
        # Longer fingers that reach ratio 2.0+ are clamped to 0 by max().
        return float(max(0.0, min(1.0, (1.8 - ratio) / 0.8)))


class HandsTracker(BaseTracker):
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        gesture_threshold: float = 0.65,
        send_landmarks: bool = False,
        send_tips: bool = False,
        send_fingers: bool = True,
        send_gestures: bool = True,
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
        self._send_landmarks = send_landmarks
        self._send_tips      = send_tips
        self._send_fingers   = send_fingers
        self._send_gestures  = send_gestures
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

        # ── Assign each hand to a fixed slot by identity ─────────────────────
        # slot 0 = left hand   slot 1 = right hand
        # A lone right hand still goes to slot 1; a lone left hand to slot 0.
        # This way OSC addresses never change meaning between frames.
        detected_slots: set[int] = set()
        detected: list[tuple[Any, float, int]] = []  # (lms, side_float, slot)
        for raw_i, hand_lms in enumerate(result.hand_landmarks):
            side = 0.0
            if result.handedness and raw_i < len(result.handedness):
                label = result.handedness[raw_i][0].category_name.lower()
                side = 1.0 if label == "right" else 0.0
            slot = int(side)  # left=0, right=1 — always, even when alone
            detected.append((hand_lms, side, slot))
            detected_slots.add(slot)

        for hand_lms, side, i in detected:
            _DRAW.draw_landmarks(frame, hand_lms, _CONNECTIONS, _DRAW_LM, _DRAW_CN)

            pairs.append((f"/webcam/hand/{i}/side", side))

            # ── raw landmarks ────────────────────────────────────────────────
            if self._send_landmarks:
                coords = [val for lm in hand_lms for val in (lm.x, lm.y, lm.z)]
                pairs.append((f"/webcam/hand/{i}/landmarks", coords))

            # ── fingertip positions ──────────────────────────────────────────
            if self._send_tips:
                for tip_id, name in ((4, "thumb"), (8, "index"), (12, "middle"), (16, "ring"), (20, "pinky")):
                    lm = hand_lms[tip_id]
                    pairs.append((f"/webcam/hand/{i}/tip/{name}/x", float(lm.x)))
                    pairs.append((f"/webcam/hand/{i}/tip/{name}/y", float(lm.y)))

            # ── per-finger curl + open (all 5 including thumb) ───────────────
            # /curl: 0=extended  1=curled
            # /open: 0=curled    1=extended  ← map to volume/brightness
            # /open/on: 1 when finger is up
            curls = {f: _finger_curl(hand_lms, f) for f in ("thumb", "index", "middle", "ring", "pinky")}
            if self._send_fingers:
                for fname, curl in curls.items():
                    open_val = 1.0 - curl
                    pairs.append((f"/webcam/hand/{i}/finger/{fname}/curl",    curl))
                    pairs.append((f"/webcam/hand/{i}/finger/{fname}/open",    open_val))
                    pairs.append((f"/webcam/hand/{i}/finger/{fname}/open/on", 1.0 if open_val >= thr else 0.0))

            c_idx = curls["index"]
            c_mid = curls["middle"]
            c_rng = curls["ring"]
            c_pky = curls["pinky"]

            if self._send_gestures:
                # ── pinch ────────────────────────────────────────────────────
                proximity   = 1.0 - min(1.0, _dist3(hand_lms[_THUMB_TIP], hand_lms[_INDEX_TIP]) / _PINCH_MAX)
                others_open = 1.0 - (c_mid + c_rng + c_pky) / 3.0
                pinch       = float(min(proximity, others_open))

                # ── composite gestures ───────────────────────────────────────
                fist  = (c_idx + c_mid + c_rng + c_pky) / 4.0
                open_ = 1.0 - fist
                point = min(1.0 - c_idx, (c_mid + c_rng + c_pky) / 3.0)
                peace = min(1.0 - c_idx, 1.0 - c_mid, (c_rng + c_pky) / 2.0)

                for gname, gval in {"fist": fist, "open": open_, "point": point, "peace": peace, "pinch": pinch}.items():
                    gval = float(max(0.0, min(1.0, gval)))
                    pairs.append((f"/webcam/hand/{i}/gesture/{gname}",    gval))
                    pairs.append((f"/webcam/hand/{i}/gesture/{gname}/on", 1.0 if gval >= thr else 0.0))

        # ── zero out slots for hands not detected this frame ──────────────────
        all_slots = set(range(self._max_num_hands))
        for i in all_slots - detected_slots:
            pairs.append((f"/webcam/hand/{i}/side", 0.0))
            if self._send_tips:
                for name in ("thumb", "index", "middle", "ring", "pinky"):
                    pairs.append((f"/webcam/hand/{i}/tip/{name}/x", 0.0))
                    pairs.append((f"/webcam/hand/{i}/tip/{name}/y", 0.0))
            if self._send_fingers:
                for name in ("thumb", "index", "middle", "ring", "pinky"):
                    pairs.append((f"/webcam/hand/{i}/finger/{name}/curl",    0.0))
                    pairs.append((f"/webcam/hand/{i}/finger/{name}/open",    0.0))
                    pairs.append((f"/webcam/hand/{i}/finger/{name}/open/on", 0.0))
            if self._send_gestures:
                for gname in ("fist", "open", "point", "peace", "pinch"):
                    pairs.append((f"/webcam/hand/{i}/gesture/{gname}",    0.0))
                    pairs.append((f"/webcam/hand/{i}/gesture/{gname}/on", 0.0))

        return pairs

    def release(self):
        self._landmarker.close()
