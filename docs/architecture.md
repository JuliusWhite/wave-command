# Wave Command Architecture

## Goals and safety boundary

Wave Command separates hand recognition, gesture interpretation, action mapping, and webcam visualization. Thumb-to-index and thumb-to-middle pinches trigger left and right clicks. Open-hand wrist swipes to the left select the previous browser tab; swipes to the right select the next tab. Scrolling and close-window gestures are temporarily disabled. Cursor movement is not implemented.

The action mappings are `PINCH_LEFT_CLICK` → `LEFT_CLICK`, `PINCH_RIGHT_CLICK` → `RIGHT_CLICK`, `NEXT_TAB` → Ctrl+Tab, and `PREVIOUS_TAB` → Ctrl+Shift+Tab. Browser interactions are the first-iteration focus; the project direction remains broader computer interaction.

## Modules

### Recognition service

- `src/recognition/hand_detector.py` owns vision backend loading and frame-to-landmark conversion.
- If `models/hand-pose.pt` exists, it loads the optional Ultralytics YOLO model. Its pose head must return exactly 21 landmarks for each hand.
- Otherwise, MediaPipe Hands supplies the same 21-point layout. Up to two hands can be detected and drawn.
- Each `Keypoint` contains its integer `id`, pixel coordinates `x` and `y`, confidence `conf`, and normalized image coordinates `normalized_x` and `normalized_y`.
- `src/recognition/gesture_engine.py` evaluates geometry only; it has no camera, model, or OS-control responsibilities.
- The rules measure pixel-space distances from thumb tip 4 to index tip 8 and middle tip 12, each divided by the wrist-to-middle-MCP distance (0 to 9). The ratio is scale-independent and avoids aspect-ratio distortion from measuring in normalized image coordinates. A ratio at or below `PINCH_THRESHOLD` (0.45) produces the corresponding left- or right-click state. If both ratios are within the threshold, the smaller ratio must lead by at least `PINCH_DOMINANCE_MARGIN` (0.08); otherwise the pose is treated as ambiguous.

### Translation service

`src/translation/command_translator.py` maps pinch states to mouse buttons and swipe states to browser tab shortcuts. It acts only on a transition into a gesture, so holding a pose does not repeatedly trigger an action. Pinch clicks are suppressed until `COMMAND_COOLDOWN_SEC` (1.5 seconds) has elapsed; all tab switches share a 1-second cooldown. Runtime input uses PyAutoGUI.

### Webcam and visualization service

`src/webcam/visualizer.py` owns the OpenCV camera lifecycle, calls a supplied frame-processing function, draws all landmarks and skeleton connections, and renders the HUD. The thumb-to-finger line changes color when either pinch is recognized. FPS, backend, distance ratio, gesture state, and the most recent action alert are shown in the video window. Press **Q** to exit.

### Application entry point

`main.py` wires the services together. For each frame, it requests detections, evaluates the first detected hand (or `IDLE` if none), passes the state to the translator, and returns a `FrameState` to the visualizer. The detector is closed even when the camera loop exits with an error.

### Setup script

`scripts/setup_models.py` creates `models/`, initializes MediaPipe Hands to verify the default fallback, and optionally loads `models/hand-pose.pt`. It intentionally does not download an arbitrary third-party checkpoint: the repository does not assume an unverified YOLO model is a 21-keypoint hand-pose model. Do not use standard COCO pose weights, which predict 17 body keypoints. The project can run immediately with MediaPipe after installing the requirements.

## Data flow

```text
OpenCV camera frame (BGR)
        |
        v
HandDetector.detect(frame)
        |  list[HandPrediction], each with 21 Keypoint values
        v
GestureEngine.evaluate(first hand or None)
        |  GestureResult(state, pinch_distance_ratio)
        v
CommandTranslator.process(state)
        | mouse click or tab shortcut and action message
        v
Visualizer draws landmarks, skeleton, pinch line, state, and alert
```

The detector may return multiple hands; the visualizer draws all of them, while the initial gesture and translation flow uses the first detection only.

## 21 hand keypoints

The index convention follows MediaPipe Hands:

| Index | Landmark | Index | Landmark |
|---:|---|---:|---|
| 0 | Wrist | 11 | Middle finger DIP |
| 1 | Thumb CMC | 12 | Middle finger tip |
| 2 | Thumb MCP | 13 | Ring finger MCP |
| 3 | Thumb IP | 14 | Ring finger PIP |
| 4 | Thumb tip | 15 | Ring finger DIP |
| 5 | Index finger MCP | 16 | Ring finger tip |
| 6 | Index finger PIP | 17 | Pinky MCP |
| 7 | Index finger DIP | 18 | Pinky PIP |
| 8 | Index finger tip | 19 | Pinky DIP |
| 9 | Middle finger MCP | 20 | Pinky tip |
| 10 | Middle finger PIP | | |

`HAND_CONNECTIONS` in `src/config.py` lists the bone segments used to draw the hand. Left-click pinch uses tip indices 4 and 8; right-click pinch uses 4 and 12. Point 0 and point 9 define the reference hand size.

## Extending gesture recognition

1. Add a gesture state to `GestureState`.
2. Implement its geometric rule in `GestureEngine.evaluate`, using confidence-checked landmarks and a scale-normalized distance or angle.
3. Add the gesture-to-action entry and mouse-button mapping to `CommandTranslator`.
4. Add or update tests for both the geometric rule and transition/cooldown behavior.
5. Update the HUD or architecture documentation if the gesture needs extra visualization or configuration.

Thresholds, camera ID, connections, and model paths are centralized in `src/config.py`.
