import socket
import sys
import threading
import time
import webbrowser

import cv2
import uvicorn
import yaml

from app_state import AppState
from camera import Camera
from models_setup import ensure_models
from osc_client import OscClient
from trackers.base import BaseTracker


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


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = load_config(config_path)

    app_state = AppState(cfg)

    # ── web UI ──────────────────────────────────────────────────────────────
    ui_port = cfg.get("ui", {}).get("port", 8080)
    start_web_server(app_state, ui_port)
    lan_ip = get_lan_ip()
    print("webcam-osc starting")
    print(f"  Web UI  -> http://localhost:{ui_port}  (LAN: http://{lan_ip}:{ui_port})")
    # Open browser automatically after server has had time to bind
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{ui_port}")).start()

    # ── initial build ────────────────────────────────────────────────────────
    working_cfg = app_state.get_config()
    osc = OscClient(
        host=working_cfg["osc"]["host"],
        port=working_cfg["osc"]["port"],
    )
    print(f"  OSC     -> {working_cfg['osc']['host']}:{working_cfg['osc']['port']}")
    print(f"  Camera  -> index {working_cfg['camera']['index']}")
    print("  Trackers:")
    ensure_models(working_cfg.get("trackers", {}))
    trackers = build_trackers(working_cfg)

    show_preview = working_cfg.get("preview", {}).get("enabled", True)
    window_name  = working_cfg.get("preview", {}).get("window_name", "webcam-osc")

    cam_cfg = working_cfg["camera"]
    cam = Camera(
        index=cam_cfg.get("index", 0),
        width=cam_cfg.get("width", 640),
        height=cam_cfg.get("height", 480),
        fps=cam_cfg.get("fps", 30),
    )

    print("  Press 'q' in the preview window (or Ctrl-C) to quit.\n")

    prev_frame  = None
    frame_count = 0
    t_start     = time.perf_counter()

    try:
        while True:
            if app_state.shutdown_requested():
                print("Shutdown requested from browser.")
                break

            # ── hot-swap on config change ────────────────────────────────────
            if app_state.consume_dirty():
                new_cfg = app_state.get_config()

                # Re-create OSC client if host/port changed
                if (new_cfg["osc"]["host"] != working_cfg["osc"]["host"] or
                        new_cfg["osc"]["port"] != working_cfg["osc"]["port"]):
                    osc = OscClient(new_cfg["osc"]["host"], new_cfg["osc"]["port"])
                    print(f"  OSC -> {new_cfg['osc']['host']}:{new_cfg['osc']['port']}")

                # Re-create camera if index or resolution changed
                nc, oc = new_cfg["camera"], working_cfg["camera"]
                if (nc.get("index") != oc.get("index") or
                        nc.get("width") != oc.get("width") or
                        nc.get("height") != oc.get("height")):
                    cam.release()
                    cam = Camera(
                        index=nc.get("index", 0),
                        width=nc.get("width", 640),
                        height=nc.get("height", 480),
                        fps=nc.get("fps", 30),
                    )
                    prev_frame = None
                    print(f"  Camera -> index {nc.get('index', 0)}")

                # Rebuild trackers
                release_trackers(trackers)
                print("  Trackers (reloading):")
                ensure_models(new_cfg.get("trackers", {}))
                trackers = build_trackers(new_cfg)

                working_cfg = new_cfg

            # ── frame loop ───────────────────────────────────────────────────
            frame = cam.read()
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

            all_pairs: list = []
            for tracker in trackers:
                pairs = tracker.process(frame, prev_frame)
                all_pairs.extend(pairs)

            # Mirror AFTER tracker processing: MediaPipe saw the real frame
            # (correct left/right handedness), preview shows natural mirror view.
            frame = cv2.flip(frame, 1)

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

            prev_frame   = frame.copy()
            frame_count += 1

            if show_preview:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        release_trackers(trackers)
        if show_preview:
            cv2.destroyAllWindows()
        print(f"\nDone. {frame_count} frames processed.")


if __name__ == "__main__":
    main()
