# Wave Command

Wave Command is a hands-free computer interaction project. It uses a webcam to recognize hand gestures and map them to computer actions, with a long-term goal of supporting control across applications and the operating system.

## First iteration

This iteration focuses on a small set of common actions, prioritizing web-browser use. Each action has its own hand pose, so the gestures do not trigger each other:

| Pose | Movement | Action |
|---|---|---|
| Pinch-ready claw: ring and pinky curled, index and middle relaxed and slightly bent | Move the hand | Move the cursor; pinch from this pose to click |
| Upright open palm facing the camera | Move up or down | Scroll (hand up scrolls down, hand down scrolls up) |
| Index and middle fingers extended, ring and pinky curled | Swipe sideways | Switch browser tab (Ctrl+Tab / Ctrl+Shift+Tab) |
| Only the index finger extended | Swipe sideways | Browser back / forward (Alt+Left / Alt+Right) |
| Thumb tip to index tip | Pinch | Left-click |
| Thumb tip to middle tip | Pinch | Right-click |

Swipe directions are measured in the (unmirrored) camera image: right-to-left means previous tab or back, and left-to-right means next tab or forward.

- **Cursor:** see [Moving the cursor](#moving-the-cursor) below.
- **Pinches outside pointing** must be armed by showing an upright open palm first, and each pinch uses up the arming. The pinch threshold is `0.45` times the wrist-to-middle-MCP hand size, so clicks register before the fingertips touch. If both pinch ratios are within the threshold but differ by less than `0.08`, the pose is ambiguous and does not click.
- **Scrolling** starts after about `0.15` hand sizes of vertical movement. It disarms pinches: release the open palm and present it again before pinching.
- **Tab switching** fires once per two-finger pose; release the pose before the next switch, so the return stroke does not switch back.
- **Cooldowns:** clicks 1.5 s; tab and history navigation 1 s. After a tab switch, clicks and history are also paused for 1 s.

Actions are printed to the terminal and shown on the video HUD. Scrolling acts on the window under the cursor. See [docs/recognition-and-gestures.md](docs/recognition-and-gestures.md#cursor-control) for the full pointing walkthrough. The close-window gesture is temporarily disabled. These boundaries do not limit the broader project direction.

## Moving the cursor

1. **Make the hand shape.** Fold your **ring finger and pinky** down into your palm. Keep your **index and middle fingers** up but **slightly bent**, not straight (straight is the tab-switch "V"). Let the thumb rest loosely to the side, as if about to grab a small ball with three fingers.

   ```text
      index  middle
        \    /        <- slightly curved, not stiff
         \  /
     thumb  [ring + pinky folded down]
   ```

2. **Hold it still for a moment** (5 frames, about 0.2 seconds) facing the camera. The **Pointer** line in the video window counts `arming 1/5 … 5/5`, then shows `moving cursor`.
3. **Move your whole hand**, keeping the shape. The cursor follows the centre of your palm, like sliding a mouse on a desk: hand right moves the cursor right, hand up moves it up. Slow movement gives small, precise steps; quick movement crosses the screen.
4. **Click** by touching your thumb to the index fingertip (left-click) or the middle fingertip (right-click). No open palm is needed first. The cursor freezes just before the click (`frozen for click`), so it clicks where you aimed.
5. **Reposition** like lifting a mouse: drop the hand out of view, bring it back somewhere comfortable, and point again. The cursor does not jump.
6. **Stop** by opening your hand. The cursor stays where it is, and the other gestures work again. Scrolling and swipes are off while pointing.

**Emergency stop:** move the physical mouse into any screen corner; PyAutoGUI's fail-safe stops the app.

If the cursor does not move, check the **Pointer** line: `off` means the shape is not recognized (curl the ring and pinky further), `arming` that never completes means the shape is unsteady, and `frozen for click` means the thumb is too close to a fingertip. See [Development and testing](docs/development-and-testing.md#the-cursor-does-not-move) for tuning.

## Requirements

- Python 3.10–3.12
- A webcam
- Windows, macOS, or Linux

MediaPipe provides the default 21-landmark hand detector, so the webcam flow works without downloading custom model weights. If a compatible 21-keypoint YOLO hand-pose model is available, install the optional dependencies with `python -m pip install -r requirements-yolo.txt` (this pulls in Ultralytics and PyTorch) and place the weights at `models/hand-pose.pt`; the detector will use it instead. Standard COCO pose models have 17 body keypoints and are not compatible.

## Setup

From the repository root, create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux, use `python3 -m venv .venv` and `source .venv/bin/activate`.

Install dependencies and verify the model backend:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/setup_models.py
```

`setup_models.py` creates the `models/` directory, verifies MediaPipe Hands, and checks for optional local YOLO weights. It does not download unverified weights.

## Run

```powershell
python main.py
```

Allow webcam access if prompted. The live view displays hand landmarks, skeletal connections, the current and raw gesture, the pointer state, whether pinches are armed, whether the index-only and two-finger poses are detected, finger-curl and pinch ratios, FPS, and action alerts. Press **Q** while the video window is focused to quit.

## Project layout

- `src/recognition/` — hand landmark detection and geometric gesture rules.
- `src/translation/` — gesture-to-action mapping and cursor control.
- `src/webcam/` — camera capture and visualization.
- `scripts/setup_models.py` — backend and optional weights verification.
- `requirements.txt` / `requirements-yolo.txt` — base dependencies / optional YOLO backend.
- `tests/` — unit tests; no webcam or real input needed.
- `docs/` — full documentation, starting at [docs/README.md](docs/README.md).

Run the unit tests with:

```powershell
python -m unittest discover -s tests
```
