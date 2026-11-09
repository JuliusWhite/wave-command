# Development and testing

## Repository map

```text
main.py                         Application wiring
src/config.py                   Shared thresholds and camera/model paths
src/recognition/                 Model wrapper and geometric gesture rules
src/translation/                 Gesture-to-action mapping (dry-run only)
src/webcam/                      Camera capture and drawing/HUD
scripts/setup_models.py          Verify MediaPipe and optional YOLO weights
tests/                           Unit tests for gestures and translation
models/                          Optional local YOLO checkpoint and training output
docs/                            Project documentation
.session-notes/                  Local-only session notes; ignored by Git
```

## Run and test

Start the application:

```powershell
python main.py
```

Run unit tests:

```powershell
python -m unittest discover -s tests -v
```

The tests cover the pinch ratio, idle/missing-hand behavior, translator edge-trigger behavior, and cooldown suppression. They do not require a webcam or run model inference.

## Configuration

Edit `src/config.py` for camera ID, model path, confidence threshold, pinch ratio, cooldown, HUD duration, and skeleton connections. Keep behavior shared between services in configuration rather than duplicating magic values.

## Troubleshooting

### Camera will not open

Check that the camera is connected and not held by another application. If a different camera is needed, update `CAMERA_ID` and restart.

### No YOLO file in `models/`

That is expected when running with MediaPipe. The model file is optional. `scripts/setup_models.py` checks the MediaPipe installation and an existing YOLO model; it does not create, train, or download a YOLO checkpoint.

### YOLO is selected but has the wrong keypoint count

Use a checkpoint trained for exactly 21 hand landmarks. Standard COCO pose models predict 17 body landmarks and cannot be used for this pinch rule. The detector falls back to MediaPipe if the inference result has a different keypoint shape.

### No simulated alert appears

Check that landmarks 4 and 8 are detected confidently and the pinch ratio shown in the HUD is at or below `PINCH_THRESHOLD`. The translator emits on a transition into pinch, not every frame; it also applies `COMMAND_COOLDOWN_SEC`.

## Session notes before commits

Before each commit, add or update a dated entry in the local `.session-notes/` folder. Record:

1. The goal and a summary of changes.
2. Important decisions or behavior changes.
3. Files changed, including generated model/data files if relevant.
4. Tests or validation commands run and their outcomes.
5. Remaining work or known limitations.

Use `.session-notes/TEMPLATE.md` as a starting point. After committing, optionally add the commit hash to the entry. This directory is ignored by Git and is not included in commits or pushed to GitHub. It is local-only, **not encrypted** or access-controlled; do not store credentials or other sensitive material there.

