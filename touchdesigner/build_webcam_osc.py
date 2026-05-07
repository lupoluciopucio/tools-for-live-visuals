# webcam_osc — TouchDesigner Component Builder
#
# HOW TO USE (one-time setup per machine):
#   1. Open TouchDesigner
#   2. Open Textport:  Alt+W  (or Windows menu → Textport)
#   3. In the Textport, type exactly this (update the path if needed):
#        exec(open('/Users/lucandrea/Documents/GitHub/tools-for-live-visuals/touchdesigner/build_webcam_osc.py').read())
#      OR just paste the whole file content and press Enter
#   4. A 'webcam_osc' Base COMP appears at the root of your network
#   5. Right-click it → Save Component → save as webcam_osc.tox
#      (e.g. ~/Documents/TouchDesigner/components/webcam_osc.tox)
#
# EVERY FUTURE PROJECT:
#   Drag webcam_osc.tox into any .toe — done.
#   Set Port / UI Port custom params, start webcam-osc, everything appears.
#
# WHAT IT CREATES (inside the Base COMP):
#
#   videostreaminTOP  ←─ MJPEG  http://127.0.0.1:UIPort/video
#       └─ out_video (null TOP)   ← annotated camera image, wire anywhere
#
#   oscinCHOP  ←─ listens on OSC Port (expr-linked to custom par)
#       │
#       ├─ select_hands → out_hands    /webcam/hand/**
#       ├─ select_face  → out_face     /webcam/face/**
#       ├─ select_pose  → out_pose     /webcam/pose/**
#       ├─ select_flow  → out_flow     /webcam/flow/**
#       └─────────────── out_all       (everything unfiltered)
#
# REFERENCING ELSEWHERE IN TD:
#   op('webcam_osc/out_video')           ← annotated webcam TOP
#   op('webcam_osc/out_hands')['pinch']
#   op('webcam_osc/out_face')['mouth_open']
#   op('webcam_osc/out_flow')['magnitude']
#
# KEY OSC CHANNELS (from webcam-osc/README.md):
#   /webcam/hand/0/pinch              float  index-thumb distance 0-1
#   /webcam/hand/0/tip/index/x        float  normalised 0-1
#   /webcam/face/mouth_open           float  0-1
#   /webcam/face/brow_raise           float  0-1
#   /webcam/face/head_tilt            float  degrees
#   /webcam/pose/left_wrist/x         float  normalised 0-1
#   /webcam/flow/magnitude            float  mean optical flow magnitude
#   /webcam/flow/direction            float  dominant angle 0-360 deg


def build_webcam_osc():
    root = op('/')

    # Remove previous instance cleanly when rebuilding
    existing = root.op('webcam_osc')
    if existing:
        existing.destroy()

    # ── Base COMP ─────────────────────────────────────────────────────────────
    b = root.create(baseCOMP, 'webcam_osc')
    b.nodeX, b.nodeY = 0, 0
    b.color = (0.08, 0.40, 0.55)
    b.comment = 'webcam-osc  |  github.com/lupoluciopucio/tools-for-live-visuals'

    # ── Custom parameters ─────────────────────────────────────────────────────
    page = b.appendCustomPage('webcam-osc')

    p_port = page.appendInt('Port', label='OSC Port')[0]
    p_port.default   = 9000
    p_port.val       = 9000
    p_port.min       = 1024
    p_port.max       = 65535
    p_port.clampMin  = True
    p_port.clampMax  = True
    p_port.help      = 'Must match osc.port in webcam-osc/config.yaml'

    p_active = page.appendToggle('Active', label='Active')[0]
    p_active.default = True
    p_active.val     = True
    p_active.help    = 'Pause OSC reception without stopping webcam-osc'

    p_uiport = page.appendInt('Uiport', label='UI Port')[0]
    p_uiport.default  = 8080
    p_uiport.val      = 8080
    p_uiport.min      = 1024
    p_uiport.max      = 65535
    p_uiport.clampMin = True
    p_uiport.clampMax = True
    p_uiport.help     = 'Must match ui.port in webcam-osc/config.yaml (default 8080)'

    # ── Video stream TOP (annotated MJPEG from webcam-osc web server) ─────────
    vstop = b.create(videostreaminTOP, 'videostream1')
    vstop.par.url.expr   = "'http://127.0.0.1:' + str(parent().par.Uiport) + '/video'"
    vstop.nodeX, vstop.nodeY = -400, 380
    vstop.comment = 'Annotated webcam feed — landmarks drawn by Python'

    out_video = b.create(outTOP, 'out_video')
    out_video.setInputs([vstop])
    out_video.nodeX, out_video.nodeY = -150, 380
    out_video.comment = 'Annotated camera image — TOP output connector'

    # ── OSC In CHOP ───────────────────────────────────────────────────────────
    oscin = b.create(oscinCHOP, 'oscin1')
    oscin.par.port.expr   = "parent().par.Port"
    oscin.par.active.expr = "parent().par.Active"
    oscin.nodeX, oscin.nodeY = -400, 0
    oscin.comment = 'Receives all /webcam/* channels'

    # ── Per-tracker Select → Null pairs ───────────────────────────────────────
    trackers = [
        ('hands', '*hand*',   250),
        ('face',  '*face*',    80),
        ('pose',  '*pose*',   -90),
        ('flow',  '*flow*',  -260),
    ]

    for name, pattern, y in trackers:
        sel = b.create(selectCHOP, 'select_' + name)
        sel.par.chop      = oscin.path
        sel.par.channames = pattern
        sel.nodeX         = -150
        sel.nodeY         = y
        sel.comment       = pattern

        null = b.create(outCHOP, 'out_' + name)
        null.setInputs([sel])
        null.nodeX   = 100
        null.nodeY   = y
        null.comment = 'CHOP output connector'

    # out_all: raw, everything (useful for discovering new channels)
    null_all = b.create(outCHOP, 'out_all')
    null_all.setInputs([oscin])
    null_all.nodeX   =  100
    null_all.nodeY   = -430
    null_all.comment = 'All /webcam/* channels unfiltered — CHOP output connector'

    # ── Info Text DAT ─────────────────────────────────────────────────────────
    txt = b.create(textDAT, 'info')
    txt.text = "\n".join([
        "webcam-osc component",
        "─────────────────────────────────────",
        "Start the Python app:",
        "  cd webcam-osc",
        "  uv run python main.py",
        "",
        "Outputs (TOP):",
        "  out_video  annotated webcam image",
        "",
        "Outputs (CHOP):",
        "  out_hands  /webcam/hand/**",
        "  out_face   /webcam/face/**",
        "  out_pose   /webcam/pose/**",
        "  out_flow   /webcam/flow/**",
        "  out_all    everything",
        "",
        "Example expressions:",
        "  op('webcam_osc/out_video')             # TOP",
        "  op('webcam_osc/out_hands')['pinch']",
        "  op('webcam_osc/out_face')['mouth_open']",
        "  op('webcam_osc/out_flow')['magnitude']",
    ])
    txt.nodeX, txt.nodeY = -400, -280

    print("webcam_osc created at /")
    print("  out_video TOP   <- annotated webcam (MJPEG from Python)")
    print("  out_hands/face/pose/flow CHOPs <- OSC data")
    print("Right-click the node -> Save Component... -> webcam_osc.tox")


build_webcam_osc()
