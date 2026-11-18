# Wave Command

Wave Command is a hands-free computer interaction project. It uses a webcam to recognize hand gestures and map them to computer actions, with a long-term goal of supporting control across applications and the operating system.

## First iteration

This iteration focuses on a small set of common actions, prioritizing web-browser use. Each action has its own hand pose, so the gestures do not trigger each other:

| Pose | Movement | Action |
|---|---|---|
| Upright open palm facing the camera | Move up or down | Scroll (hand up scrolls down, hand down scrolls up) |
| Index and middle fingers extended, ring and pinky curled | Swipe sideways | Switch browser tab (Ctrl+Tab / Ctrl+Shift+Tab) |
| Only the index finger extended | Swipe sideways | Browser back / forward (Alt+Left / Alt+Right) |
| Thumb tip to index tip | Pinch | Left-click |
| Thumb tip to middle tip | Pinch | Right-click |

Swipe directions are measured in the (unmirrored) camera image: right-to-left means previous tab or back, and left-to-right means next tab or forward.

- **Pinches** must be armed by showing an upright open palm first, and each pinch uses up the arming. The pinch threshold is `0.45` times the wrist-to-middle-MCP hand size, so clicks register before the fingertips touch. If both pinch ratios are within the threshold but differ by less than `0.08`, the pose is ambiguous and does not click.
- **Scrolling** starts after about `0.15` hand sizes of vertical movement. It disarms pinches: release the open palm and present it again before pinching.
- **Tab switching** fires once per two-finger pose; release the pose before the next switch, so the return stroke does not switch back.
- **Cooldowns:** clicks 1.5 s; tab and history navigation 1 s. After a tab switch, clicks and history are also paused for 1 s.

Actions are printed to the terminal and shown on the video HUD. Cursor movement is not included in this iteration, so position the cursor manually over content before clicking or scrolling. The close-window gesture is temporarily disabled. These boundaries do not limit the broader project direction.

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

Allow webcam access if prompted. The live view displays hand landmarks, skeletal connections, the current and raw gesture, whether pinches are armed, whether the index-only and two-finger poses are detected, finger-curl and pinch ratios, FPS, and action alerts. Press **Q** while the video window is focused to quit.

## Project layout

- `src/recognition/` — hand landmark detection and geometric gesture rules.
- `src/translation/` — gesture-to-mouse-action mapping.
- `src/webcam/` — camera capture and visualization.
- `scripts/setup_models.py` — backend and optional weights verification.
- `requirements.txt` / `requirements-yolo.txt` — base dependencies / optional YOLO backend.
- `tests/` — unit tests; no webcam or real input needed.
- `docs/` — full documentation, starting at [docs/README.md](docs/README.md).

Run the unit tests with:

```powershell
python -m unittest discover -s tests
```
