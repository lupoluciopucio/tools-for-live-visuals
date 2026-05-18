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
#   Set custom params (Port, UI Port, Mode, Tracking File, Video File).
#
# TWO MODES (custom par "Mode"):
#   0 = Live   — OSC from Python app + MJPEG webcam stream
#   1 = Baked  — JSON + pre-rendered tracked video (frame-locked, no Python needed)
#
# BAKED WORKFLOW:
#   1. In the webcam-osc web UI, upload a video → "Pre-process & Export"
#   2. Download the _tracked.mp4 and _tracking.json files
#   3. Set Mode → Baked, point Trackingfile + Videofile to those files
#   4. out_video and all CHOP outputs are now frame-locked to TD's timeline
#   5. Offline export (Realtime OFF) works perfectly — no OSC lag
#
# OUTPUTS (same in both modes):
#   out_video               TOP  — annotated video
#   out_hands               CHOP — /webcam/hand/** (all)
#   out_hands_gestures      CHOP — gesture channels
#   out_hands_fingers       CHOP — finger curl/open channels
#   out_hands_tips          CHOP — fingertip x/y channels
#   out_hands_landmarks     CHOP — raw landmark coords
#   out_face / out_pose / out_flow / out_all
#
# REFERENCING:
#   op('webcam_osc/out_video')
#   op('webcam_osc/out_hands_gestures')['hand_0_gesture_fist']   # Baked
#   op('webcam_osc/out_hands_gestures')['/webcam/hand/0/gesture/fist']  # Live


# ── Script CHOP cook code (Baked mode) ────────────────────────────────────────
# Reads _tracking.json, maps TD frame → video frame, outputs all channels.
# Embedded as a string so the builder can write it into a Text DAT.
_SCRIPT_CHOP_CODE = '''
import json

# Module-level state
_cache     = {}      # {path: data}  — cached JSON
_vid_frame = 0.0     # current playhead position (float, sub-frame accumulation)

def cook(scriptOp):
    global _cache, _vid_frame

    scriptOp.clear()

    # Touch me.time.frame — tells TD this CHOP depends on time, so it re-cooks every frame
    _tick   = me.time.frame
    td_fps  = project.cookRate

    play    = int(parent().par.Play)
    restart = int(parent().par.Restart)

    if restart:
        _vid_frame = 0.0

    path = str(parent().par.Trackingfile)
    if not path:
        scriptOp.appendChan("playhead")[0] = 0.0
        return

    if path not in _cache:
        try:
            with open(path) as f:
                data = json.load(f)
            _cache = {path: data}
            _vid_frame = 0.0   # reset when a new file is loaded
        except Exception as e:
            debug("tracking_reader load error:", str(e))
            scriptOp.appendChan("playhead")[0] = 0.0
            return

    data        = _cache[path]
    channels    = data.get("channels", {})
    video_fps   = data.get("fps", 30.0)
    frame_count = data.get("frame_count", 1)

    # Advance internal playhead by the fps ratio each cook (pauses cleanly when play=0)
    if play:
        _vid_frame += video_fps / td_fps
        if _vid_frame >= frame_count:
            _vid_frame = 0.0   # loop

    vid_frame = max(0, min(int(_vid_frame), frame_count - 1))

    for address, values in channels.items():
        name = address.lstrip("/").replace("/", "_")
        if address.startswith("/webcam/"):
            name = address[len("/webcam/"):].replace("/", "_")
        ch = scriptOp.appendChan(name)
        ch[0] = float(values[vid_frame]) if vid_frame < len(values) else 0.0

    # Shared playhead channel — moviefileinTOP tracks this to stay in sync
    scriptOp.appendChan("playhead")[0] = float(vid_frame)
'''


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

    p_mode = page.appendMenu('Mode', label='Mode')[0]
    p_mode.menuNames   = ['live', 'baked']
    p_mode.menuLabels  = ['Live (OSC + webcam)', 'Baked (JSON + video)']
    p_mode.default     = 0
    p_mode.val         = 0
    p_mode.help        = 'Live: receive OSC from Python app. Baked: read pre-processed JSON + video file.'

    p_trackfile = page.appendFile('Trackingfile', label='Tracking JSON')[0]
    p_trackfile.help = 'Path to _tracking.json exported by webcam-osc Pre-process & Export'

    p_vidfile = page.appendFile('Videofile', label='Tracked Video')[0]
    p_vidfile.help = 'Path to _tracked.mp4 exported by webcam-osc Pre-process & Export'

    p_play = page.appendToggle('Play', label='Play')[0]
    p_play.default = True
    p_play.val     = True
    p_play.help    = 'Play / Pause baked video + tracking'

    p_restart = page.appendPulse('Restart', label='Restart')[0]
    p_restart.help = 'Jump back to frame 0'

    p_sync = page.appendFloat('Syncoffset', label='Sync Offset (frames)')[0]
    p_sync.default  = 0.0
    p_sync.val      = 0.0
    p_sync.min      = -60.0
    p_sync.max      = 60.0
    p_sync.clampMin = True
    p_sync.clampMax = True
    p_sync.help     = 'Nudge video frame relative to tracking (+ = video ahead, - = video behind)'

    # ── LIVE: Video stream TOP (annotated MJPEG from webcam-osc web server) ───
    vstop = b.create(videostreaminTOP, 'live_videostream')
    vstop.par.url.expr   = "'http://127.0.0.1:' + str(parent().par.Uiport) + '/video'"
    vstop.nodeX, vstop.nodeY = -600, 420
    vstop.comment = 'LIVE: annotated webcam feed via MJPEG'

    # ── BAKED: Movie File In TOP ───────────────────────────────────────────────
    movtop = b.create(moviefileinTOP, 'baked_video')
    movtop.par.file.expr      = "parent().par.Videofile"
    movtop.par.playmode       = 'specify'           # Specify Index
    movtop.par.indexunit      = 'indices'           # raw frame number — fps-independent
    movtop.par.index.expr     = "op('tracking_reader')['playhead'] + parent().par.Syncoffset"
    movtop.par.index.mode     = ParMode.EXPRESSION
    movtop.nodeX, movtop.nodeY = -600, 340
    movtop.comment = 'BAKED: frame-locked to tracking_reader playhead (indices unit)'

    # ── Switch TOP (Live ↔ Baked) ──────────────────────────────────────────────
    sw_top = b.create(switchTOP, 'switch_video')
    sw_top.setInputs([vstop, movtop])
    sw_top.par.index.expr = "parent().par.Mode"
    sw_top.nodeX, sw_top.nodeY = -350, 380
    sw_top.comment = '0=Live  1=Baked'

    out_video = b.create(outTOP, 'out_video')
    out_video.setInputs([sw_top])
    out_video.nodeX, out_video.nodeY = -100, 380
    out_video.comment = 'Annotated video — TOP output connector'

    # ── LIVE: OSC In CHOP ─────────────────────────────────────────────────────
    oscin = b.create(oscinCHOP, 'oscin1')
    oscin.par.port.expr   = "parent().par.Port"
    oscin.par.active.expr = "parent().par.Active"
    oscin.nodeX, oscin.nodeY = -600, 0
    oscin.comment = 'LIVE: receives all /webcam/* channels via OSC'

    # ── BAKED: Script CHOP (reads _tracking.json) ────────────────────────────
    script_dat = b.create(textDAT, 'tracking_reader_script')
    script_dat.text = _SCRIPT_CHOP_CODE
    script_dat.nodeX, script_dat.nodeY = -600, -200
    script_dat.comment = 'Cook script for tracking_reader Script CHOP'

    script_chop = b.create(scriptCHOP, 'tracking_reader')
    script_chop.par.callbacks = script_dat   # pass operator object, not string
    script_chop.setInputs([oscin])           # dummy input — forces re-cook every frame
    script_chop.nodeX, script_chop.nodeY = -600, -130
    script_chop.comment = 'BAKED: per-frame tracking values from JSON'

    # ── Switch CHOP helper: merge Live + Baked then pick by Mode ─────────────
    # We create a Switch CHOP for each output group so patchers never need to
    # rewire. Each Switch takes [live_select, baked_select] and is driven by Mode.

    def make_output_pair(name, pattern, node_y):
        """
        Create: live_select_<name> → switch_<name> ← baked_select_<name>
                                           ↓
                                      out_<name>
        """
        # Live side: select from oscin
        live_sel = b.create(selectCHOP, 'live_select_' + name)
        live_sel.par.chop      = 'oscin1'
        live_sel.par.channames = pattern
        live_sel.nodeX         = -350
        live_sel.nodeY         = node_y + 60
        live_sel.comment       = 'LIVE: ' + pattern

        # Baked side: select from script_chop (same pattern, but chan names use _)
        baked_sel = b.create(selectCHOP, 'baked_select_' + name)
        baked_sel.par.chop      = 'tracking_reader'
        baked_sel.par.channames = pattern.replace('/', '_').replace('*', '*')
        baked_sel.nodeX         = -350
        baked_sel.nodeY         = node_y - 30
        baked_sel.comment       = 'BAKED: ' + pattern

        # Switch CHOP
        sw = b.create(switchCHOP, 'switch_' + name)
        sw.setInputs([live_sel, baked_sel])
        sw.par.index.expr = "parent().par.Mode"
        sw.nodeX, sw.nodeY = -100, node_y
        sw.comment = '0=Live  1=Baked'

        # Output connector
        out = b.create(outCHOP, 'out_' + name)
        out.setInputs([sw])
        out.nodeX   = 150
        out.nodeY   = node_y
        out.comment = 'CHOP output connector'

    tracker_groups = [
        ('hands', '*hand*',   250),
        ('face',  '*face*',    80),
        ('pose',  '*pose*',   -90),
        ('flow',  '*flow*',  -260),
    ]
    for name, pattern, y in tracker_groups:
        make_output_pair(name, pattern, y)

    hand_subtypes = [
        ('hands_gestures',  '*hand*gesture*',   -370),
        ('hands_fingers',   '*hand*finger*',    -450),
        ('hands_tips',      '*hand*tip*',       -530),
        ('hands_landmarks', '*hand*landmark*',  -610),
    ]
    for name, pattern, y in hand_subtypes:
        make_output_pair(name, pattern, y)

    # out_all: raw live OSC (no baked equivalent — use out_hands/face etc. in Baked mode)
    null_all = b.create(outCHOP, 'out_all')
    null_all.setInputs([oscin])
    null_all.nodeX   =  150
    null_all.nodeY   = -700
    null_all.comment = 'All live /webcam/* channels unfiltered (Live mode only)'

    # ── Info Text DAT ─────────────────────────────────────────────────────────
    txt = b.create(textDAT, 'info')
    txt.text = "\n".join([
        "webcam-osc component  —  two modes",
        "─────────────────────────────────────────────",
        "Mode = Live  (default)",
        "  Start Python app:  cd webcam-osc && uv run python main.py",
        "  Outputs driven by live OSC + MJPEG stream",
        "",
        "Mode = Baked",
        "  1. In the webcam-osc web UI, upload a video",
        "  2. Click 'Pre-process & Export'",
        "  3. Download _tracked.mp4 and _tracking.json",
        "  4. Set Tracking JSON + Tracked Video params",
        "  5. All outputs are now frame-locked (works with offline export)",
        "",
        "Outputs (TOP):",
        "  out_video              annotated video (Live: MJPEG / Baked: MP4)",
        "",
        "Outputs (CHOP) — hands:",
        "  out_hands              /webcam/hand/**  (all)",
        "  out_hands_gestures     gesture channels",
        "  out_hands_fingers      finger curl/open channels",
        "  out_hands_tips         fingertip x/y channels",
        "  out_hands_landmarks    raw landmark coords",
        "",
        "  hand/0 = LEFT   hand/1 = RIGHT",
        "",
        "Outputs (CHOP) — other trackers:",
        "  out_face / out_pose / out_flow / out_all (Live only)",
        "",
        "Channel name format in Baked mode:",
        "  /webcam/hand/0/gesture/fist  →  hand_0_gesture_fist",
        "",
        "Example expressions:",
        "  op('webcam_osc/out_video')",
        "  op('webcam_osc/out_hands_gestures')['hand_0_gesture_fist']  # Baked",
        "  op('webcam_osc/out_hands_fingers')['hand_0_finger_index_open']  # Baked",
    ])
    txt.nodeX, txt.nodeY = -600, -430

    print("webcam_osc created at /")
    print("  Mode par: 0=Live (OSC+webcam)  1=Baked (JSON+video)")
    print("  out_video TOP  |  out_hands / out_hands_gestures / ... CHOPs")
    print("Right-click the node -> Save Component... -> webcam_osc.tox")


build_webcam_osc()
