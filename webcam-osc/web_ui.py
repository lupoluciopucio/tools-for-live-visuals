import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app_state import AppState

_STATIC = Path(__file__).parent / "static"
_UPLOAD_DIR = Path(__file__).parent / "uploads"

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

# Injected by main.py before server starts
state: AppState | None = None


@app.get("/", response_class=HTMLResponse)
async def index():
    return (_STATIC / "index.html").read_text()


@app.get("/api/config")
async def get_config():
    return JSONResponse(state.get_config())


@app.post("/api/config")
async def post_config(patch: dict):
    state.update_config(patch)
    return JSONResponse({"ok": True})


def _probe_cameras() -> list[dict]:
    """Blocking call — probe indices and get macOS camera names. Run in executor."""
    mac_names: list[str] = []
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["system_profiler", "SPCameraDataType", "-json"],
                capture_output=True, text=True, timeout=4,
            ).stdout
            data = json.loads(out)
            # system_profiler lists external/virtual cameras first on M1;
            # reverse so index 0 (built-in FaceTime) stays at position 0
            names_raw = [c.get("_name", "") for c in data.get("SPCameraDataType", [])]
            mac_names = list(reversed(names_raw))
        except Exception:
            pass

    cameras = []
    for idx in range(4):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            if idx < len(mac_names) and mac_names[idx]:
                name = mac_names[idx]
            elif idx == 0:
                name = "Built-in Camera"
            else:
                name = f"External Camera {idx}"
            preferred = any(k in name.lower() for k in ("facetime", "built-in", "builtin", "hd camera", "isight"))
            cameras.append({"index": idx, "name": name, "preferred": preferred})
            cap.release()
    if cameras and not any(c["preferred"] for c in cameras):
        cameras[0]["preferred"] = True
    return cameras


@app.get("/api/cameras")
async def get_cameras():
    loop = asyncio.get_event_loop()
    cameras = await loop.run_in_executor(None, _probe_cameras)
    return JSONResponse(cameras)


@app.post("/api/shutdown")
async def shutdown():
    state.request_shutdown()
    return JSONResponse({"ok": True})


@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)):
    """Save an uploaded video file and switch the source to it."""
    _UPLOAD_DIR.mkdir(exist_ok=True)
    dest = _UPLOAD_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    state.reset_video_state()
    state.update_config({"camera": {"video_path": str(dest)}})
    return JSONResponse({"ok": True, "name": file.filename, "path": str(dest)})


@app.post("/api/video/control")
async def video_control(body: dict):
    """Set play/pause, loop, or clear the video source."""
    if "playing" in body:
        state.set_video_playing(bool(body["playing"]))
    if "loop" in body:
        state.set_video_loop(bool(body["loop"]))
    if body.get("action") == "clear":
        state.update_config({"camera": {"video_path": None}})
    return JSONResponse({"ok": True})


@app.get("/api/video/state")
async def video_state():
    """Current video playback state (for UI sync on page load)."""
    cfg = state.get_config()
    return JSONResponse({
        "video_path": cfg.get("camera", {}).get("video_path"),
        "playing": state.get_video_playing(),
        "loop": state.get_video_loop(),
    })


@app.get("/video")
async def video_feed():
    """MJPEG stream of the annotated webcam feed."""
    async def generate():
        while True:
            frame = state.get_frame()
            if frame is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            await asyncio.sleep(0.05)  # ~20 fps cap
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws/osc")
async def osc_websocket(ws: WebSocket):
    await ws.accept()
    last_seq = 0
    try:
        while True:
            new_seq, entries = state.get_osc_log_since(last_seq)
            if entries:
                last_seq = new_seq
                await ws.send_text(json.dumps(entries))
            await asyncio.sleep(0.05)  # ~20 fps push rate
    except WebSocketDisconnect:
        pass
