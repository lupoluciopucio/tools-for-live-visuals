import copy
import threading
from collections import deque
from typing import Any


class AppState:
    """
    Thread-safe shared state between the main camera loop and the web UI server.
    """

    def __init__(self, config: dict):
        self._lock = threading.Lock()
        self._config: dict = copy.deepcopy(config)
        self._dirty = False
        self._shutdown = False
        # Ring buffer of recent OSC messages: (seq, address, value_repr)
        self._osc_seq: int = 0
        self._osc_log: deque[tuple[int, str, str]] = deque(maxlen=200)
        # Latest JPEG-encoded frame for the web preview
        self._frame_jpeg: bytes | None = None
        self._frame_lock = threading.Lock()
        # Video playback controls (ephemeral — not persisted in config)
        self._video_playing: bool = True
        self._video_loop: bool = True
        # Batch pre-processing state
        self._preprocess: dict = {
            "status":    "idle",   # "idle" | "processing" | "done" | "error"
            "progress":  0.0,      # 0.0 – 1.0
            "video_out": None,
            "json_out":  None,
            "error":     None,
        }

    # ------------------------------------------------------------------ config

    def get_config(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._config)

    def update_config(self, patch: dict):
        """Deep-merge patch into the current config and mark dirty."""
        with self._lock:
            _deep_merge(self._config, patch)
            self._dirty = True

    def consume_dirty(self) -> bool:
        """Returns True (and clears the flag) if config changed since last call."""
        with self._lock:
            if self._dirty:
                self._dirty = False
                return True
            return False

    def request_shutdown(self):
        with self._lock:
            self._shutdown = True

    def shutdown_requested(self) -> bool:
        with self._lock:
            return self._shutdown

    # ---------------------------------------------------------------- osc log

    def set_frame(self, jpeg_bytes: bytes):
        with self._frame_lock:
            self._frame_jpeg = jpeg_bytes

    def get_frame(self) -> bytes | None:
        with self._frame_lock:
            return self._frame_jpeg

    def log_osc(self, address: str, value: Any):
        val_repr = _compact(value)
        with self._lock:
            self._osc_seq += 1
            self._osc_log.append((self._osc_seq, address, val_repr))

    def get_osc_log_since(self, last_seq: int) -> tuple[int, list[tuple[str, str]]]:
        """Returns (new_seq, [(address, value), ...]) with entries newer than last_seq."""
        with self._lock:
            new = [(addr, val) for seq, addr, val in self._osc_log if seq > last_seq]
            return self._osc_seq, new

    # -------------------------------------------------------------- video

    def get_video_playing(self) -> bool:
        with self._lock:
            return self._video_playing

    def set_video_playing(self, playing: bool):
        with self._lock:
            self._video_playing = playing

    def get_video_loop(self) -> bool:
        with self._lock:
            return self._video_loop

    def set_video_loop(self, loop: bool):
        with self._lock:
            self._video_loop = loop

    def reset_video_state(self):
        """Called when a new video is loaded."""
        with self._lock:
            self._video_playing = True
            self._video_loop = True

    # ---------------------------------------------------------- batch preprocess

    def set_preprocess_state(
        self,
        status: str,
        progress: float = 0.0,
        video_out: str | None = None,
        json_out: str | None = None,
        error: str | None = None,
    ):
        with self._lock:
            self._preprocess = {
                "status":    status,
                "progress":  progress,
                "video_out": video_out,
                "json_out":  json_out,
                "error":     error,
            }

    def get_preprocess_state(self) -> dict:
        with self._lock:
            return dict(self._preprocess)


# ------------------------------------------------------------------ helpers

def _deep_merge(base: dict, patch: dict):
    for key, val in patch.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


def _compact(value: Any) -> str:
    if isinstance(value, list):
        if len(value) > 6:
            return f"[{', '.join(f'{v:.3f}' if isinstance(v, float) else str(v) for v in value[:6])}  … {len(value)} vals]"
        return str([round(v, 3) if isinstance(v, float) else v for v in value])
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
