# Wave Command Architecture

## Goals and safety boundary

Wave Command separates hand recognition, gesture interpretation, action mapping, and webcam visualization. The first iteration is a dry run: the translator emits an action label to the terminal, and the webcam HUD displays it. No module sends keyboard or mouse input or executes operating-system commands. `PyAutoGUI` and `pynput` are included as dependencies for future work but are not imported by the runtime.

The current action mapping is `PINCH_CLOSE_WINDOW` → `CLOSE_WINDOW (Simulated)`. Browser interactions are the first-iteration focus; the project direction remains broader computer interaction.

## Modules

### Recognition service

- `src/recognition/hand_detector.py` owns vision backend loading and frame-to-landmark conversion.
- If `models/hand-pose.pt` exists, it loads the optional Ultralytics YOLO model. Its pose head must return exactly 21 landmarks for each hand.
- Otherwise, MediaPipe Hands supplies the same 21-point layout. Up to two hands can be detected and drawn.
- Each `Keypoint` contains its integer `id`, pixel coordinates `x` and `y`, confidence `conf`, and normalized image coordinates `normalized_x` and `normalized_y`.
- `src/recognition/gesture_engine.py` evaluates geometry only; it has no camera, model, or OS-control responsibilities.
- The initial rule measures the pixel-space distance between keypoints 4 and 8 and divides it by the wrist-to-middle-MCP distance (0 to 9). The ratio is scale-independent and avoids aspect-ratio distortion from measuring in normalized image coordinates. A ratio at or below `PINCH_THRESHOLD` (0.35) produces `PINCH_CLOSE_WINDOW`.

### Translation service

`src/translation/command_translator.py` maps gesture states to action labels. It emits only on the transition into a pinch, so holding a pinch does not repeatedly trigger the action. Further transitions are suppressed until `COMMAND_COOLDOWN_SEC` (1.5 seconds) has elapsed. The only output is a formatted simulated message; there is deliberately no command-execution implementation.

### Webcam and visualization service

`src/webcam/visualizer.py` owns the OpenCV camera lifecycle, calls a supplied frame-processing function, draws all landmarks and skeleton connections, and renders the HUD. The thumb-to-index line changes color when the recognized state is a pinch. FPS, backend, distance ratio, gesture state, and the most recent simulated alert are shown in the video window. Press **Q** to exit.

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
        |  optional simulated alert string; never executes it
        v
Visualizer draws landmarks, skeleton, pinch line, state, and alert
```

The detector may return multiple hands; the visualizer draws all of them, while the initial gesture/translation flow uses the first detection only.

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

`HAND_CONNECTIONS` in `src/config.py` lists the bone segments used to draw the hand. The pinch uses tip indices 4 and 8; point 0 and point 9 define the reference hand size.

## Extending gesture recognition

1. Add a gesture state to `GestureState`.
2. Implement its geometric rule in `GestureEngine.evaluate`, using confidence-checked landmarks and a scale-normalized distance or angle.
3. Add the gesture-to-action entry to `CommandTranslator.ACTIONS`. Keep translation limited to a dry-run label until real command execution is explicitly designed and reviewed.
4. Add or update tests for both the geometric rule and transition/cooldown behavior.
5. Update the HUD or architecture documentation if the gesture needs extra visualization or configuration.

Thresholds, camera ID, connections, and model paths are centralized in `src/config.py`.
