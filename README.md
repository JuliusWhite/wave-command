# Wave Command

Wave Command is a hands-free computer interaction project. It uses a webcam to recognize hand gestures and map them to computer actions, with a long-term goal of supporting control across applications and the operating system.

## First iteration

This iteration focuses on a small set of common actions, prioritizing web-browser use. A thumb-to-index pinch triggers a left-click, and a thumb-to-middle-finger pinch triggers a right-click. With an open hand, swiping from the right side of the camera view to the left selects the previous browser tab; swiping left-to-right selects the next tab. Pinch clicks have a 1.5-second cooldown, and tab switches have a 1-second cooldown. Actions are printed to the terminal and shown on the video HUD. Scrolling and close-window gestures are temporarily disabled.

Cursor movement is not included in this iteration, so position the cursor manually over content before clicking. A horizontal wrist swipe must move at least `1.25` hand sizes to switch tabs. The pinch threshold is `0.45` times the wrist-to-middle-MCP hand size, allowing clicks before the fingertips touch. If both pinch ratios are within the threshold but differ by less than `0.08`, the pose is treated as ambiguous and does not click. These boundaries do not limit the broader project direction.

## Requirements

- Python 3.10–3.12
- A webcam
- Windows, macOS, or Linux

MediaPipe provides the default 21-landmark hand detector, so the webcam flow works without downloading custom model weights. If a compatible 21-keypoint YOLO hand-pose model is available, place it at `models/hand-pose.pt`; the detector will use it instead. Standard COCO pose models have 17 body keypoints and are not compatible.

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

Allow webcam access if prompted. The live view displays hand landmarks, skeletal connections, gesture and pinch diagnostics, FPS, and action alerts. Press **Q** while the video window is focused to quit.

## Project layout

- `src/recognition/` — hand landmark detection and geometric gesture rules.
- `src/translation/` — gesture-to-mouse-action mapping.
- `src/webcam/` — camera capture and visualization.
- `scripts/setup_models.py` — backend and optional weights verification.
- `docs/architecture.md` — architecture, data flow, and landmark reference.

Run the unit tests with:

```powershell
python -m unittest discover -s tests
```
