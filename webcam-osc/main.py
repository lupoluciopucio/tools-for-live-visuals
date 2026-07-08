import socket
import sys
import threading
import time
import webbrowser

import cv2
import numpy as np
import uvicorn
import yaml

from app_state import AppState
from camera import Camera, VideoFileSource
from models_setup import ensure_models
from osc_client import OscClient
from trackers.base import BaseTracker


def _letterbox(frame: np.ndarray, tw: int, th: int) -> np.ndarray:
    """Fit frame into (tw × th) preserving aspect ratio; pad with black bars."""
    h, w = frame.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((th, tw, 3), dtype=np.uint8)
    x, y = (tw - nw) // 2, (th - nh) // 2
    out[y:y + nh, x:x + nw] = resized
    return out


def _fit_resize(frame: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """Scale frame to fit within (max_w × max_h) preserving aspect ratio. No padding."""
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h)
    if abs(scale - 1.0) < 0.01:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def release_trackers(trackers: list[BaseTracker]):
    for t in trackers:
        t.release()


def start_web_server(app_state: AppState, ui_port: int):
    import web_ui
    web_ui.state = app_state
    config = uvicorn.Config(
        web_ui.app,
        host="0.0.0.0",
        port=ui_port,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server


def get_lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def build_trackers(cfg: dict) -> list[BaseTracker]:
    trackers: list[BaseTracker] = []
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
            thumb_curl_sensitivity=h.get("thumb_curl_sensitivity", 1.5),
            send_absent_zeros=h.get("send_absent_zeros", False),
        ))
        print("  [+] HandsTracker")

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
        print("  [+] FaceTracker")

    if tc.get("pose", {}).get("enabled", False):
        from trackers.pose import PoseTracker
        p = tc["pose"]
        trackers.append(PoseTracker(
            min_detection_confidence=p.get("min_detection_confidence", 0.5),
            min_tracking_confidence=p.get("min_tracking_confidence", 0.5),
            model_complexity=p.get("model_complexity", 1),
        ))
        print("  [+] PoseTracker")

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
        print("  [+] OpticalFlowTracker")

    return trackers


def main(
    force_no_preview: bool = False,
    _app_state: "AppState | None" = None,
    config_path: str | None = None,
):
    if config_path is None:
        config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = load_config(config_path)

    app_state = _app_state if _app_state is not None else AppState(cfg)

    # ── web UI ──────────────────────────────────────────────────────────────
    ui_port = cfg.get("ui", {}).get("port", 8080)
    start_web_server(app_state, ui_port)
    lan_ip = get_lan_ip()
    print("webcam-osc starting")
    print(f"  Web UI  -> http://localhost:{ui_port}  (LAN: http://{lan_ip}:{ui_port})")
    # Open browser automatically (skip when launched via menubar – it handles this)
    if _app_state is None:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{ui_port}")).start()

    # ── initial build ────────────────────────────────────────────────────────
    working_cfg = app_state.get_config()
    osc = OscClient(
        host=working_cfg["osc"]["host"],
        port=working_cfg["osc"]["port"],
    )
    print(f"  OSC     -> {working_cfg['osc']['host']}:{working_cfg['osc']['port']}")
    print("  Trackers:")
    ensure_models(working_cfg.get("trackers", {}))
    trackers = build_trackers(working_cfg)

    show_preview = (not force_no_preview) and working_cfg.get("preview", {}).get("enabled", True)
    window_name  = working_cfg.get("preview", {}).get("window_name", "webcam-osc")

    cam_cfg = working_cfg["camera"]
    _video_path = cam_cfg.get("video_path")
    if _video_path:
        source = VideoFileSource(_video_path)
        source_fps: float | None = source.fps
        print(f"  Source  -> video: {_video_path}")
    else:
        source = Camera(
            index=cam_cfg.get("index", 0),
            width=cam_cfg.get("width", 640),
            height=cam_cfg.get("height", 480),
            fps=cam_cfg.get("fps", 30),
        )
        source_fps = None
        print(f"  Camera  -> index {cam_cfg.get('index', 0)}")

    prev_frame       = None
    prev_track_frame = None
    frame_count = 0
    t_start     = time.perf_counter()

    try:
        while True:
            if app_state.shutdown_requested():
                print("Shutdown requested from browser.")
                break

            if app_state.get_paused():
                time.sleep(0.1)
                continue

            # ── hot-swap on config change ────────────────────────────────────
            if app_state.consume_dirty():
                new_cfg = app_state.get_config()

                # Re-create OSC client if host/port changed
                if (new_cfg["osc"]["host"] != working_cfg["osc"]["host"] or
                        new_cfg["osc"]["port"] != working_cfg["osc"]["port"]):
                    osc = OscClient(new_cfg["osc"]["host"], new_cfg["osc"]["port"])
                    print(f"  OSC -> {new_cfg['osc']['host']}:{new_cfg['osc']['port']}")

                # Re-create source if video path or camera params changed
                nc, oc = new_cfg["camera"], working_cfg["camera"]
                new_vp = nc.get("video_path")
                old_vp = oc.get("video_path")
                source_needs_rebuild = (
                    new_vp != old_vp or
                    (not new_vp and (
                        nc.get("index") != oc.get("index") or
                        nc.get("width") != oc.get("width") or
                        nc.get("height") != oc.get("height")
                    ))
                )
                if source_needs_rebuild:
                    source.release()
                    if new_vp:
                        source = VideoFileSource(new_vp)
                        source_fps = source.fps
                        print(f"  Source -> video: {new_vp}")
                    else:
                        source = Camera(
                            index=nc.get("index", 0),
                            width=nc.get("width", 640),
                            height=nc.get("height", 480),
                            fps=nc.get("fps", 30),
                        )
                        source_fps = None
                        print(f"  Source -> camera index {nc.get('index', 0)}")
                    prev_frame = None
                    prev_track_frame = None

                # Rebuild trackers
                release_trackers(trackers)
                print("  Trackers (reloading):")
                ensure_models(new_cfg.get("trackers", {}))
                trackers = build_trackers(new_cfg)

                working_cfg = new_cfg

            # ── frame loop ───────────────────────────────────────────────────
            frame_t = time.perf_counter()
            _is_video = bool(working_cfg["camera"].get("video_path"))

            if _is_video:
                _playing = app_state.get_video_playing()
                _looping = app_state.get_video_loop()
                if not _playing:
                    frame = source.last_frame()
                    if frame is None:
                        time.sleep(0.033)
                        continue
                else:
                    # Skip frames when processing is slower than video FPS
                    # so playback stays real-time rather than falling behind.
                    if source_fps and source_fps > 0:
                        _budget = 1.0 / source_fps
                        _debt = time.perf_counter() - frame_t
                        while _debt > _budget * 1.5:
                            _skip = source.read()
                            if _skip is None:
                                break
                            _debt -= _budget

                    frame = source.read()
                    if frame is None:  # EOF
                        if _looping:
                            source.seek_start()
                            frame = source.read()
                        if frame is None:
                            app_state.set_video_playing(False)
                            frame = source.last_frame()
                            if frame is None:
                                time.sleep(0.033)
                                continue

                # Fit to display bounds preserving aspect ratio.
                # Auto-swap width/height for portrait content so vertical
                # space is used instead of adding side bars.
                _dw = working_cfg["camera"].get("width", 640)
                _dh = working_cfg["camera"].get("height", 480)
                _fh, _fw = frame.shape[:2]
                if _fh > _fw:  # portrait video — use a taller bounding box
                    _dw, _dh = min(_dw, _dh), max(_dw, _dh)
                frame = _fit_resize(frame, _dw, _dh)
            else:
                frame = source.read()
                if frame is None:
                    print("Camera read failed — exiting.")
                    break

            rot = working_cfg.get("camera", {}).get("rotation", 0)
            if rot == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rot == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # ── track at reduced resolution ────────────────────────────────────
            # Compute tracking dims by preserving the display frame's aspect
            # ratio within the configured tracking budget. This handles both
            # landscape (640×480 → 320×240) and portrait (360×640 → 180×320)
            # content without squishing.
            _disw = frame.shape[1]
            _dish = frame.shape[0]
            _tc   = working_cfg.get("tracking", {})
            _max_side = max(_tc.get("width", 320), _tc.get("height", 240))
            _t_scale  = min(_max_side / _disw, _max_side / _dish)
            _trw = int(_disw * _t_scale)
            _trh = int(_dish * _t_scale)
            if _trw != _disw or _trh != _dish:
                track_frame = cv2.resize(frame, (_trw, _trh), interpolation=cv2.INTER_LINEAR)
            else:
                track_frame = frame

            all_pairs: list = []
            for tracker in trackers:
                pairs = tracker.process(track_frame, prev_track_frame)
                all_pairs.extend(pairs)

            # Mirror only for live camera — video files are already oriented
            if not _is_video:
                track_frame = cv2.flip(track_frame, 1)

            # Scale annotated track_frame back up for the web preview
            if _trw != _disw or _trh != _dish:
                frame = cv2.resize(track_frame, (_disw, _dish), interpolation=cv2.INTER_LINEAR)
            else:
                frame = track_frame

            osc.send_pairs(all_pairs)

            # Log to OSC monitor (sample every 3rd frame to keep UI light)
            if frame_count % 3 == 0:
                for address, value in all_pairs:
                    app_state.log_osc(address, value)

            # Encode annotated frame for web UI preview (~20 fps is enough)
            elapsed = time.perf_counter() - t_start
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}  OSC :{working_cfg['osc']['port']}  UI :{ui_port}",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
            )
            if frame_count % 2 == 0:  # encode every other frame (~15-20 fps)
                ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    app_state.set_frame(jpeg.tobytes())

            prev_track_frame = track_frame.copy()
            prev_frame       = frame.copy()
            frame_count += 1

            # Throttle video playback to its native FPS (camera read blocks naturally)
            if _is_video and source_fps:
                elapsed = time.perf_counter() - frame_t
                time.sleep(max(0.0, 1.0 / source_fps - elapsed))

            if show_preview:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        source.release()
        release_trackers(trackers)
        if show_preview:
            cv2.destroyAllWindows()
        print(f"\nDone. {frame_count} frames processed.")


if __name__ == "__main__":
    main()
