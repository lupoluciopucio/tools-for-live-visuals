"""
batch.py — Offline video pre-processing for webcam-osc.

Runs all enabled trackers over every frame of a video file (no real-time
throttle), writes:
  {stem}_tracked.mp4   — annotated video (landmarks drawn)
  {stem}_tracking.json — per-channel per-frame data + metadata

Usage (from Python):
    from batch import batch_process
    video_out, json_out = batch_process("/path/to/clip.mp4", cfg, progress_cb)

Usage (CLI):
    uv run python batch.py /path/to/clip.mp4
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

# ── helpers (mirrors main.py) ─────────────────────────────────────────────────

def _fit_resize(frame: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h)
    if abs(scale - 1.0) < 0.01:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)


def _build_trackers(cfg: dict):
    from models_setup import ensure_models
    ensure_models(cfg.get("trackers", {}))

    trackers = []
    tc = cfg.get("trackers", {})

    if tc.get("hands", {}).get("enabled", False):
        from trackers.hands import HandsTracker
        h = tc["hands"]
        trackers.append(HandsTracker(
            max_num_hands=h.get("max_num_hands", 2),
            min_detection_confidence=h.get("min_detection_confidence", 0.7),
            min_tracking_confidence=h.get("min_tracking_confidence", 0.5),
            gesture_threshold=h.get("gesture_threshold", 0.65),
            send_landmarks=h.get("send_landmarks", False),
            send_tips=h.get("send_tips", False),
            send_fingers=h.get("send_fingers", True),
            send_gestures=h.get("send_gestures", True),
        ))

    if tc.get("face", {}).get("enabled", False):
        from trackers.face import FaceTracker
        f = tc["face"]
        trackers.append(FaceTracker(
            max_num_faces=f.get("max_num_faces", 1),
            min_detection_confidence=f.get("min_detection_confidence", 0.5),
            min_tracking_confidence=f.get("min_tracking_confidence", 0.5),
            send_raw_landmarks=f.get("send_raw_landmarks", True),
            landmark_chunk_size=f.get("landmark_chunk_size", 200),
        ))

    if tc.get("pose", {}).get("enabled", False):
        from trackers.pose import PoseTracker
        p = tc["pose"]
        trackers.append(PoseTracker(
            min_detection_confidence=p.get("min_detection_confidence", 0.5),
            min_tracking_confidence=p.get("min_tracking_confidence", 0.5),
            model_complexity=p.get("model_complexity", 1),
        ))

    if tc.get("optical_flow", {}).get("enabled", False):
        from trackers.optical_flow import OpticalFlowTracker
        of = tc["optical_flow"]
        trackers.append(OpticalFlowTracker(
            pyr_scale=of.get("pyr_scale", 0.5),
            levels=of.get("levels", 3),
            winsize=of.get("winsize", 15),
            iterations=of.get("iterations", 3),
            poly_n=of.get("poly_n", 5),
            poly_sigma=of.get("poly_sigma", 1.2),
            grid_enabled=of.get("grid_enabled", False),
            grid_size=of.get("grid_size", 8),
        ))

    return trackers


# ── main batch function ───────────────────────────────────────────────────────

def batch_process(
    video_path: str,
    cfg: dict,
    progress_cb: Callable[[int, int], None] | None = None,
) -> tuple[str, str]:
    """
    Process every frame of *video_path* with the enabled trackers.

    Returns (annotated_video_path, tracking_json_path).
    Calls progress_cb(current_frame, total_frames) each frame if provided.
    Raises on any fatal error (caller should catch and update state).
    """
    src = Path(video_path)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output paths — next to source file
    video_out = str(src.parent / (src.stem + "_tracked.mp4"))
    json_out  = str(src.parent / (src.stem + "_tracking.json"))

    # Determine display size (portrait-aware, same logic as main.py)
    disp_w = cfg.get("camera", {}).get("width",  640)
    disp_h = cfg.get("camera", {}).get("height", 480)
    if src_h > src_w:          # portrait — use taller box
        disp_w, disp_h = min(disp_w, disp_h), max(disp_w, disp_h)

    # Tracking resolution (smaller = faster, same logic as main.py)
    _tc      = cfg.get("tracking", {})
    _max_side = max(_tc.get("width", 320), _tc.get("height", 240))

    # Video writer — match display size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_out, fourcc, fps, (disp_w, disp_h))

    trackers = _build_trackers(cfg)

    # Per-channel accumulator: address → list of float values per frame
    channels: dict[str, list[float]] = {}

    prev_track_frame = None
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Fit to display size (portrait-aware, no padding)
            frame = _fit_resize(frame, disp_w, disp_h)
            # Pad if smaller than writer expects (e.g. rounding)
            if frame.shape[1] != disp_w or frame.shape[0] != disp_h:
                canvas = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
                h, w = frame.shape[:2]
                canvas[:h, :w] = frame
                frame = canvas

            # Compute tracking resolution
            _t_scale = min(_max_side / disp_w, _max_side / disp_h)
            _trw = int(disp_w * _t_scale)
            _trh = int(disp_h * _t_scale)
            if _trw != disp_w or _trh != disp_h:
                track_frame = cv2.resize(frame, (_trw, _trh), interpolation=cv2.INTER_LINEAR)
            else:
                track_frame = frame

            # Run all trackers
            all_pairs: list[tuple[str, Any]] = []
            for tracker in trackers:
                pairs = tracker.process(track_frame, prev_track_frame)
                all_pairs.extend(pairs)

            # Scale annotated track_frame back to display size
            if _trw != disp_w or _trh != disp_h:
                frame = cv2.resize(track_frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
            else:
                frame = track_frame

            # Write annotated frame
            writer.write(frame)

            # Accumulate channel data (scalar values only — skip blob landmarks)
            for address, value in all_pairs:
                if isinstance(value, (int, float)):
                    if address not in channels:
                        channels[address] = []
                    channels[address].append(float(value))

            prev_track_frame = track_frame.copy()
            frame_idx += 1

            if progress_cb:
                progress_cb(frame_idx, total_frames)

    finally:
        cap.release()
        writer.release()
        for t in trackers:
            t.release()

    # Pad any short channels to frame_idx length (missed frames → 0.0)
    for addr in channels:
        while len(channels[addr]) < frame_idx:
            channels[addr].append(0.0)

    # Write JSON
    payload = {
        "fps":          fps,
        "frame_count":  frame_idx,
        "width":        disp_w,
        "height":       disp_h,
        "source":       str(src),
        "channels":     channels,
    }
    with open(json_out, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    return video_out, json_out


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml

    if len(sys.argv) < 2:
        print("Usage: uv run python batch.py <video_path> [config.yaml]")
        sys.exit(1)

    video_path  = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    t0 = time.perf_counter()
    last_pct = -1

    def _cli_progress(n, total):
        global last_pct
        pct = int(100 * n / total) if total else 0
        if pct != last_pct:
            last_pct = pct
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r  [{bar}] {pct:3d}%  frame {n}/{total}", end="", flush=True)

    print(f"batch-processing: {video_path}")
    vid, jsn = batch_process(video_path, cfg, _cli_progress)
    elapsed = time.perf_counter() - t0
    print(f"\n  done in {elapsed:.1f}s")
    print(f"  video → {vid}")
    print(f"  json  → {jsn}")
