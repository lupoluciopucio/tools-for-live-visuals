# tools-for-live-visuals

A collection of standalone tools for live visual performance with TouchDesigner and similar software.

## Tools

| Tool | Description |
|---|---|
| [`webcam-osc`](webcam-osc/) | Webcam analysis (hands, face, pose, optical flow) → OSC → TouchDesigner |
| [`touchdesigner`](touchdesigner/) | TD component builder script + saved `.tox` files |

## TouchDesigner quick setup

1. In the TD Textport (`Alt+W`) run:
   ```python
   exec(open('/Users/lucandrea/Documents/GitHub/tools-for-live-visuals/touchdesigner/build_webcam_osc.py').read())
   ```
2. Right-click the generated `webcam_osc` node → **Save Component...** → `webcam_osc.tox`
3. Every future project: drag `webcam_osc.tox` in, start `webcam-osc`, done.
