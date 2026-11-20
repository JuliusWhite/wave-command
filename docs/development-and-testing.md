# Development and testing

## Repository map

```text
main.py                         Application wiring
src/config.py                   Shared thresholds and camera/model paths
src/recognition/                 Model wrapper and geometric gesture rules
src/translation/                 Gesture-to-action mapping and cursor control
src/webcam/                      Camera capture, drawing/HUD, status overlay
requirements.txt                 Base runtime dependencies (MediaPipe backend)
requirements-yolo.txt            Optional Ultralytics/PyTorch for a YOLO hand model
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

The tests cover the pointer pose and session (3D claw detection with depth, arming, palm tracking, direct pinch clicks, freezing, ignored swipes, exit), the pointer controller (direction, acceleration, jitter, freeze and reposition without jumps, edge clamping), pinch classification and activation, the open-palm scroll-start threshold and scroll-to-pinch re-arming, two-finger tab swipes (including the return stroke, pinch exclusion, and a scroll-then-swipe run through the translator), index-only history swipes, gesture-confusion cases (detection glitches, stray frames of another pose, noisy and flickering pinches, moving claws, slow drift, diagonal and return strokes, stale pinch arming), shortcut/wheel mappings and cooldowns, idle/missing-hand behavior, translator edge-trigger behavior, click cooldown suppression, PyAutoGUI dispatch with its pause disabled, the HUD text, the status overlay (status mapping, running without the preview, camera toggling), hand tracking with several hands in view (locking onto the first hand, swapped detector order, short dropouts, hand-over once the first hand is gone), and bypass (button and **B** key toggling, the `BYPASSED` status, and no stale action on resume). Close-window gesture tests remain skipped while that feature is disabled. Input dispatch is injected in tests; the tests do not require a webcam, OS input, or model inference.

## Configuration

Edit `src/config.py` for camera ID, model path, confidence threshold, pinch ratio, pointer arming/freeze/filter/speed settings, scroll start distance/dead zone/smoothing/speed, tab and history swipe distances, cooldowns, multi-hand tracking (`HAND_TRACK_MAX_JUMP`, `HAND_TRACK_LOST_FRAMES`), HUD duration, status overlay and camera preview at start (`SHOW_STATUS_OVERLAY`, `SHOW_CAMERA_PREVIEW`), overlay opacity and top margin, and skeleton connections. The overlay's brand colors and icon tables are defined in `src/webcam/overlay.py`; the mode colors (shared by the HUD and the overlay) are defined in `Visualizer` in `src/webcam/visualizer.py`. Keep behavior shared between services in configuration rather than duplicating magic values.

## Troubleshooting

### Camera will not open

Check that the camera is connected and not held by another application. If a different camera is needed, update `CAMERA_ID` and restart.

### No YOLO file in `models/`

That is expected when running with MediaPipe. The model file is optional. `scripts/setup_models.py` checks the MediaPipe installation and an existing YOLO model; it does not create, train, or download a YOLO checkpoint.

### YOLO is selected but has the wrong keypoint count

Use a checkpoint trained for exactly 21 hand landmarks. Standard COCO pose models predict 17 body landmarks and cannot be used for this pinch rule. The detector falls back to MediaPipe if the inference result has a different keypoint shape.

### No click alert appears

Show an upright open palm facing the camera before each pinch to arm pinch input, then pinch within `PINCH_ARM_TIMEOUT_SEC` (3 s); the compact HUD shows `pinch armed` once it is armed, and the detailed view (**D**) shows the *Pinch gate* line. The hand must point upward and show a broad palm rather than being turned sideways. Detected scrolling disarms the open-hand activation; after scrolling, stop showing the open-hand pose and present it again to re-arm. Losing the hand also resets the gate. Then check that landmarks 4 and 8 (left-click) or 4 and 12 (right-click) are detected confidently and the relevant pinch ratio stays at or below `PINCH_THRESHOLD` for `PINCH_CONFIRM_FRAMES` (2) frames. A swipe pose cancels the arming. The translator clicks on a transition into a pinch, not every frame; it also applies `COMMAND_COOLDOWN_SEC`, and ignores clicks for 1 second after a tab switch. Place the cursor over the intended target first.

### The cursor does not move

Press **D** and check the *Pointer* line. If it stays `off`, the claw is not recognized. With MediaPipe, read the *Flex 3D I/M/R/P* line: the index and middle should average at least `POINTER_CLAW_MIN_FLEX_DEGREES` (35) and stay below `POINTER_CLAW_MAX_FLEX_DEGREES` (150), and the ring and pinky should be curled. If your claw reads below the band, lower `POINTER_CLAW_MIN_FLEX_DEGREES`; if straight fingers read above it, raise it. Keep the palm facing the camera. If it shows `arming` but never reaches `5/5`, hold the hand still: moving more than `POINTER_ARM_MAX_DRIFT_RATIO` hand sizes restarts arming, and a swipe pose must end first. If it shows `frozen for click`, the thumb is too close to the index or middle tip; move it further away.

### The cursor is too shaky, too slow, or too fast

Lower `POINTER_FILTER_MIN_CUTOFF_HZ` for less jitter at rest, or raise `POINTER_FILTER_BETA` for less lag when moving quickly. Adjust `POINTER_SLOW_PX_PER_HAND_SIZE` for precision and `POINTER_FAST_PX_PER_HAND_SIZE` for speed. If horizontal movement is reversed (for example with a mirrored camera), set `POINTER_INVERT_X = False`.

### Tab switching does not happen

Check that the compact HUD shows `TABS` (or, in the detailed view, `Two-finger tab pose: detected`). Index and middle must be straight and ring and pinky clearly curled; a relaxed ring finger often stays between the extended and curled angles, so the pose is not recognized. Keep the thumb away from the index and middle tips, because an active pinch cancels the pose. Swipe at least `TAB_SWIPE_DISTANCE_RATIO` (1.25) hand sizes sideways. Swipe briskly: the distance must be covered within `SWIPE_WINDOW_SEC` (0.8 s). After a switch, the return stroke is ignored; swipe the same way again to switch again, after the 1-second cooldown. Swipes are disabled while the HUD shows `CURSOR`; open the ring and pinky first. With MediaPipe, the index and middle must also be straight in 3D; if the HUD shows `CURSOR` or nothing instead of `TABS`, compare the *Flex 3D* values with `POINTER_CLAW_MIN_FLEX_DEGREES`.

### Scrolling does not start

Check that the HUD shows `SCROLL` while the palm is up. The palm must stay upright and face the camera with all fingers straight. Movement must be mostly vertical and exceed `SCROLL_START_DISTANCE_RATIO` (0.15) hand sizes; sideways movement restarts the measurement. Scrolling acts on the window under the cursor.

## Session notes before commits

Before each commit, add or update a dated entry in the local `.session-notes/` folder. Record:

1. The goal and a summary of changes.
2. Important decisions or behavior changes.
3. Files changed, including generated model/data files if relevant.
4. Tests or validation commands run and their outcomes.
5. Remaining work or known limitations.

Use `.session-notes/TEMPLATE.md` as a starting point. After committing, optionally add the commit hash to the entry. This directory is ignored by Git and is not included in commits or pushed to GitHub. It is local-only, **not encrypted** or access-controlled; do not store credentials or other sensitive material there.
