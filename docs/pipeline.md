# Project pipeline

Wave Command has a preparation pipeline (install dependencies and optionally provide weights) and a real-time runtime pipeline (camera frame through gesture recognition and mouse action).

## 1. Prepare the environment

From the repository root, create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To use an optional YOLO hand model, install `requirements-yolo.txt` instead; it adds Ultralytics (and PyTorch) on top of the base requirements.

On macOS or Linux, create the environment with `python3 -m venv .venv` and activate it with `source .venv/bin/activate`.

Verify the model setup:

```powershell
python scripts/setup_models.py
```

The script creates `models/`, initializes MediaPipe Hands to verify its API is available, and checks for `models/hand-pose.pt`. It does not download model weights or start training. If YOLO weights are present and Ultralytics is installed, the script attempts to load them; use a model trained for 21 hand keypoints.

## 2. Start the application

```powershell
python main.py
```

Or use the **Wave Command** shortcut created by `scripts/create_shortcut.py`, which starts `wave_command.pyw` with `pythonw.exe`: the same `main()` runs without a console, with output written to `wave-command.log`.

`main()` first sets the Windows taskbar identity (`APP_USER_MODEL_ID`), then creates a `HandDetector`, `HandTracker`, `GestureEngine`, `CommandTranslator`, `PointerController`, a `StatusOverlay` (unless `SHOW_STATUS_OVERLAY` is off or Tk is unavailable), and `Visualizer`. The visualizer opens the configured camera (`CAMERA_ID`, initially `0`) and passes each captured BGR frame to the application callback.

## 3. Per-frame runtime flow

1. **Capture:** OpenCV reads a frame from the webcam.
2. **Detect:** `HandDetector.detect(frame)` returns zero or more `HandPrediction` objects with 21 keypoints each.
   - If `models/hand-pose.pt` exists and Ultralytics can load it, YOLO is selected.
   - Otherwise, the detector selects MediaPipe Hands.
   - If YOLO inference returns a pose shape other than 21 keypoints, the detector switches to MediaPipe.
3. **Evaluate:** `HandTracker.update()` picks the tracked hand: the first hand to appear, followed by palm position until it has been gone for `HAND_TRACK_LOST_FRAMES` frames. Other hands are ignored. `main.py` gives the tracked hand (or `None`) to `GestureEngine.evaluate()`.
4. **Recognize:** The engine checks landmark confidence, calculates pinch and finger-curl ratios and (with MediaPipe depth) 3D finger flexion, debounces the movement poses with a pose stabilizer, confirms and holds left/right pinches (with looser rules for the second pinch of a double-click), tracks the palm centre while the pinch-ready claw pose is held for cursor control, and otherwise detects vertical wrist movement with an upright open hand for scrolling, horizontal movement with the index and middle fingers extended for tab switching, horizontal movement with the index, middle, and ring fingers extended for window switching, or horizontal movement with only the index finger extended for history navigation. Similar simultaneous pinch ratios are ignored as ambiguous.
   - **Bypass:** while bypass is on, steps 5 and 6 send nothing: the pointer is reset and the translator only observes the gesture, so resuming does not fire a held gesture.
5. **Point:** `PointerController.update()` smooths the palm position and moves the cursor relatively; it holds still while a pinch forms or when no sample is reported.
6. **Translate:** `CommandTranslator.process(gesture, scroll_amount)` dispatches left/right mouse clicks on new pinch transitions, subject to the 1.5-second cooldown, or a double-click for a second left pinch within `DOUBLE_CLICK_WINDOW_SEC`; mouse-wheel steps for open-palm vertical movement; Ctrl+Tab/Ctrl+Shift+Tab for two-finger swipes; Alt+Tab/Alt+Shift+Tab (with Alt held briefly) for three-finger swipes; or Alt+Left/Alt+Right for index-only swipes. Navigation shortcuts share a 1-second cooldown, and a tab or window switch also pauses clicks and history for that second. Input is issued by PyAutoGUI with its per-call pause disabled; errors are allowed to surface rather than being reported as success.
7. **Visualize:** The callback returns a `FrameState`. The visualizer applies any overlay button clicks, updates the status overlay, and, when the camera preview is shown, draws the tracked hand (ignored hands as grey skeletons), bones, pinch-tip line, state, gesture diagnostics, FPS, and a temporary alert in the camera window.
8. **Exit:** Click the overlay's **✕**, or press **Q** while the video window is focused. Camera, window, and overlay resources are released in `finally` blocks.

```text
Camera frame
  -> HandDetector
  -> 21 keypoints per hand
  -> GestureEngine
  -> IDLE, PINCH_LEFT_CLICK, PINCH_RIGHT_CLICK, SCROLL_UP/DOWN,
     NEXT_TAB/PREVIOUS_TAB, NEXT_WINDOW/PREVIOUS_WINDOW, BACK, or FORWARD
  -> PointerController (cursor) and CommandTranslator
  -> OS cursor movement, mouse clicks, wheel input, tab/window/history shortcuts, and action alert
  -> Status overlay and (optional) Visualizer HUD
```

## Output and controls

- Terminal: a message identifying the executed click, scroll, tab, window, or history action.
- Status overlay: small always-on-top row of icons, centered at the top of the screen: mode icon, last action, hand-in-view dot, and **pause**/**play** (bypass), **camera** (show/hide preview), and **✕** (quit) buttons.
- Video window (optional): hand skeleton, colored finger bones, gesture and raw state, pointer state, pinch gate, detected swipe poses, curl and pinch ratios, FPS, and alert.
- Keyboard: index-only swipes send Alt+Left or Alt+Right to navigate history; two-finger swipes send Ctrl+Tab or Ctrl+Shift+Tab; three-finger swipes send Alt+Tab or Alt+Shift+Tab; in the video window, **B** toggles bypass, **C** hides the preview, **D** toggles the detailed HUD, and **Q** quits.
- Mouse: the claw pose moves the cursor; pinches click and open-palm vertical movements scroll at the current cursor position. Moving the physical mouse into a screen corner triggers PyAutoGUI's fail-safe and stops the app.

## Model files

- `models/hand-pose.pt` — optional compatible YOLO hand checkpoint, loaded by the detector when present.
- MediaPipe's pretrained resources — supplied by the installed package and loaded by MediaPipe; they do not appear as a `.pt` file in the project folder.
- `yolo11n-pose.pt` — a standard pose checkpoint, if present at the repository root, is not loaded by this app and is not a 21-keypoint hand checkpoint.
