# webcam-osc

Standalone Python app: reads your webcam, runs computer vision trackers, and streams everything as OSC to TouchDesigner (or any OSC receiver).

Works with **any TouchDesigner version** — no plugins, no upgrades needed.

---

## Requirements

- macOS (M1/M2/Intel), Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) — install once with `curl -LsSf https://astral.sh/uv/install.sh | sh`

---

## Setup

```bash
cd webcam-osc
uv sync          # creates .venv and installs all deps (~1–2 min first time)
```

---

## Run

```bash
uv run python main.py
```

Press **`q`** in the preview window to quit.

To use a different config file:

```bash
uv run python main.py my_config.yaml
```

---

## Configuration (`config.yaml`)

```yaml
osc:
  host: "127.0.0.1"  # TD machine IP (same machine = 127.0.0.1)
  port: 9000

camera:
  index: 0            # 0 = built-in webcam, 1 = external
  width: 640
  height: 480
  fps: 30

preview:
  enabled: true       # set false to run headless (e.g. background process)

trackers:
  hands:
    enabled: true     # disable any tracker to save CPU
  face:
    enabled: true
    send_raw_landmarks: true   # false = only mouth_open, brow_raise, head_tilt
  pose:
    enabled: true
    model_complexity: 1        # 0=lite (fastest), 1=full, 2=heavy
  optical_flow:
    enabled: true
    grid_enabled: false        # enable to get per-cell motion data
    grid_size: 8
```

---

## OSC address reference

### Hands (`/webcam/hand/{i}/...`)

| Address | Type | Description |
|---|---|---|
| `/webcam/hand/{i}/landmarks` | `[float]*63` | x,y,z × 21 landmarks (normalized 0–1) |
| `/webcam/hand/{i}/pinch` | `float` | index–thumb distance (normalized) |
| `/webcam/hand/{i}/tip/thumb/x` | `float` | thumb tip X |
| `/webcam/hand/{i}/tip/thumb/y` | `float` | thumb tip Y |
| `/webcam/hand/{i}/tip/index/x` | `float` | index tip X |
| `/webcam/hand/{i}/tip/index/y` | `float` | index tip Y |
| `/webcam/hand/{i}/tip/middle/x` | `float` | — |
| `/webcam/hand/{i}/tip/ring/x` | `float` | — |
| `/webcam/hand/{i}/tip/pinky/x` | `float` | — |

### Face (`/webcam/face/...`)

| Address | Type | Description |
|---|---|---|
| `/webcam/face/mouth_open` | `float` | mouth open ratio (0–1) |
| `/webcam/face/brow_raise` | `float` | brow raise ratio (0–1) |
| `/webcam/face/head_tilt` | `float` | head roll in degrees |
| `/webcam/face/landmarks/{n}` | `[float]*200` | raw 468 landmarks chunked (x,y,z each) |

### Pose (`/webcam/pose/...`)

| Address | Type | Description |
|---|---|---|
| `/webcam/pose/landmarks` | `[float]*132` | x,y,z,visibility × 33 |
| `/webcam/pose/nose/x` | `float` | — |
| `/webcam/pose/left_shoulder/x` | `float` | — |
| `/webcam/pose/right_shoulder/x` | `float` | — |
| `/webcam/pose/left_wrist/x` | `float` | — |
| `/webcam/pose/right_wrist/x` | `float` | — |
| *(+ elbow, hip pairs)* | | |

### Optical Flow (`/webcam/flow/...`)

| Address | Type | Description |
|---|---|---|
| `/webcam/flow/magnitude` | `float` | mean Farneback flow magnitude |
| `/webcam/flow/direction` | `float` | dominant flow direction 0–360° |
| `/webcam/flow/grid/{y}/{x}` | `float` | per-cell magnitude (if `grid_enabled: true`) |

---

## TouchDesigner setup

1. Add an **OSC In CHOP**
2. Set **Network Port** to `9000` (or whatever `config.yaml` says)
3. Enable **Active**
4. Channels like `/webcam/hand/0/pinch`, `/webcam/face/mouth_open`, etc. appear automatically

For list messages (landmarks), add a **CHOP Execute DAT** or use a **Reorder CHOP** to split the flat array into per-landmark channels.

---

## Adding a new tracker

1. Create `trackers/my_tracker.py` subclassing `BaseTracker`
2. Implement `process(frame, prev_frame) -> list[tuple[str, Any]]`
3. Add an `enabled` block under `trackers:` in `config.yaml`
4. Load it in `main.py` `build_trackers()` — copy the pattern from the other trackers
