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

`main()` creates a `HandDetector`, `GestureEngine`, `CommandTranslator`, and `Visualizer`. The visualizer opens the configured camera (`CAMERA_ID`, initially `0`) and passes each captured BGR frame to the application callback.

## 3. Per-frame runtime flow

1. **Capture:** OpenCV reads a frame from the webcam.
2. **Detect:** `HandDetector.detect(frame)` returns zero or more `HandPrediction` objects with 21 keypoints each.
   - If `models/hand-pose.pt` exists and Ultralytics can load it, YOLO is selected.
   - Otherwise, the detector selects MediaPipe Hands.
   - If YOLO inference returns a pose shape other than 21 keypoints, the detector switches to MediaPipe.
3. **Evaluate:** `main.py` gives the first detected hand (or `None`) to `GestureEngine.evaluate()`.
4. **Recognize:** The engine checks landmark confidence, calculates pinch and finger-curl ratios, classifies a left/right pinch or idle hand, and detects horizontal wrist movement with an open hand as a tab-switch gesture. Similar simultaneous pinch ratios are ignored as ambiguous.
5. **Translate:** `CommandTranslator.process()` dispatches left/right mouse clicks on new pinch transitions, subject to the 1.5-second cooldown, and Ctrl+Tab or Ctrl+Shift+Tab for the corresponding open-hand swipe, subject to a shared 1-second tab-switch cooldown. Input is issued by PyAutoGUI; errors are allowed to surface rather than being reported as success.
6. **Visualize:** The callback returns a `FrameState`. The visualizer draws all detected hands, bones, pinch-tip line, state, gesture diagnostics, FPS, and a temporary alert in the camera window.
7. **Exit:** Press **Q** while the video window is focused. Camera/window resources are released in `finally` blocks.

```text
Camera frame
  -> HandDetector
  -> 21 keypoints per hand
  -> GestureEngine
  -> IDLE, PINCH_LEFT_CLICK, PINCH_RIGHT_CLICK, NEXT_TAB, or PREVIOUS_TAB
  -> CommandTranslator
  -> OS mouse clicks/tab shortcuts and action alert
  -> Visualizer HUD
```

## Output and controls

- Terminal: a message identifying the executed click or tab action.
- Video window: hand skeleton, colored finger bones, pinch state, ratio, FPS, and alert.
- Keyboard: open-hand swipes send Ctrl+Tab or Ctrl+Shift+Tab to navigate tabs; **Q** exits the visualizer when its window is focused.
- Mouse: pinches click at the current cursor position; cursor movement is not implemented.

## Model files

- `models/hand-pose.pt` — optional compatible YOLO hand checkpoint, loaded by the detector when present.
- MediaPipe's pretrained resources — supplied by the installed package and loaded by MediaPipe; they do not appear as a `.pt` file in the project folder.
- `yolo11n-pose.pt` — a standard pose checkpoint, if present at the repository root, is not loaded by this app and is not a 21-keypoint hand checkpoint.
