"""
menubar.py – macOS menu bar entry point for WebcamOSC.

Runs the camera/OSC processing loop in a background thread and puts a
persistent menu bar icon on the main thread via rumps.  Double-click
WebcamOSC.app to launch; use the menu bar icon to open the Web UI or quit.
"""

import os
import sys
import threading
import webbrowser

# ── Make sure imports resolve whether we're called from inside webcam-osc/ or
#    from the .app bundle whose CWD may be different. ────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rumps

import yaml
from app_state import AppState


def _load_ui_port(config_path: str = "config.yaml") -> int:
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("ui", {}).get("port", 8080)
    except Exception:
        return 8080


class WebcamOSCApp(rumps.App):
    def __init__(self):
        super().__init__("📷", quit_button=None)
        self._ui_port = _load_ui_port()
        self.menu = [
            rumps.MenuItem("Open Web UI", callback=self.open_ui),
            None,  # separator
            rumps.MenuItem("Quit WebcamOSC", callback=self.quit_app),
        ]
        self._app_state: AppState | None = None
        # Poll every second: if the browser triggered /api/shutdown (tab
        # closed), quit the menu bar app automatically.
        self._shutdown_timer = rumps.Timer(self._check_shutdown, 1)
        self._shutdown_timer.start()

    # ── polling ───────────────────────────────────────────────────────────────
    def _check_shutdown(self, _):
        if self._app_state is not None and self._app_state.shutdown_requested():
            rumps.quit_application()

    # ── menu callbacks ────────────────────────────────────────────────────────
    def open_ui(self, _):
        webbrowser.open(f"http://127.0.0.1:{self._ui_port}")

    def quit_app(self, _):
        if self._app_state is not None:
            self._app_state.request_shutdown()
        rumps.quit_application()


def main():
    # Pre-load config and AppState so the worker can reference the same object.
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    app_state = AppState(cfg)

    menubar_app = WebcamOSCApp()
    menubar_app._ui_port = cfg.get("ui", {}).get("port", 8080)

    # Kick off the worker now; it will import main and call main.main()
    # but we need to share the already-created app_state.  We monkey-patch
    # the main module's _shared_app_state before it calls AppState().
    import main as _main_module
    _main_module._shared_app_state = app_state
    _main_module._shared_config_path = config_path
    menubar_app._app_state = app_state

    worker = threading.Thread(
        target=lambda: _main_module.main(
            force_no_preview=True,
            _app_state=app_state,
            config_path=config_path,
        ),
        daemon=True,
        name="webcam-osc-loop",
    )
    worker.start()

    # Open browser after the server has had time to bind
    threading.Timer(3.0, lambda: webbrowser.open(f"http://127.0.0.1:{menubar_app._ui_port}")).start()

    menubar_app.run()


if __name__ == "__main__":
    main()
