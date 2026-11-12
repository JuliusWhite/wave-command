# Recognition and gestures

## Detector backend selection

`HandDetector` in `src/recognition/hand_detector.py` converts each frame into `HandPrediction` objects with a common structure:

- `id`: index in the hand's 21-point layout.
- `x`, `y`: pixel coordinates in the source image.
- `normalized_x`, `normalized_y`: image coordinates scaled to `[0, 1]`.
- `conf`: confidence associated with the point.
- `HandPrediction.confidence`: hand-level confidence.

Backend selection is based on the configured model path:

1. If `models/hand-pose.pt` exists and Ultralytics can be imported, load YOLO.
2. Otherwise, load MediaPipe Hands.
3. If YOLO inference does not expose exactly 21 keypoints, initialize MediaPipe and process the frame with that backend.

MediaPipe is ready to use after installing dependencies. Its pretrained hand models come from the MediaPipe package, so an empty project `models/` directory is expected when using that backend.

The current `scripts/setup_models.py` is a verifier, not a trainer or downloader. A YOLO checkpoint must be supplied separately and must use the same landmark ordering expected by `GestureEngine`.

## Landmark index

The ordering matches MediaPipe Hands:

| Index | Hand landmark | Index | Hand landmark |
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

`HAND_CONNECTIONS` in `src/config.py` lists the landmark pairs the visualizer connects to draw the hand skeleton.

## Pinch rules

`GestureEngine.evaluate()` requires a hand with 21 points and confidence at least `CONFIDENCE_THRESHOLD` for the wrist (0), thumb tip (4), and middle-finger MCP (9). It computes each available finger's fingertip distance from the thumb tip, normalized by hand size:

```text
hand_size = distance(point[0], point[9])
index_pinch_ratio = distance(point[4], point[8]) / hand_size
middle_pinch_ratio = distance(point[4], point[12]) / hand_size
```

When an index-tip ratio is at or below `PINCH_THRESHOLD`, the state is `PINCH_LEFT_CLICK`. When a middle-tip ratio is at or below the threshold, the state is `PINCH_RIGHT_CLICK`. If both are below threshold, the smaller ratio wins. The ratio is based on pixel coordinates, and normalizing by hand size reduces sensitivity to how close the hand is to the camera. A missing hand, a low-confidence required landmark, or an invalid/near-zero reference distance produces `IDLE`.

A fist is maintained when at least three of the four non-thumb fingertips are closer to the middle-finger MCP (the palm reference point) than their PIP joints, with the corresponding landmarks meeting the confidence threshold. To start a fist gesture, all confidently visible fingers must be curled for three consecutive frames; this prevents a partially opened hand from triggering a close. A fully curled fist takes precedence over a pinch, while a pinch can win over a partial curl only when its pinched finger is not curled. After activation, the fist remains active until it is absent for six consecutive frames, then requires three neutral frames before another gesture can trigger. The gesture requests closure of the currently focused window on Windows and sends Alt+F4 on other platforms. The close gesture bypasses the click cooldown, and the webcam loop continues processing after the gesture.

The HUD shows the raw classifier state, per-finger curl ratios (index/middle/ring/pinky), and both pinch ratios alongside the filtered gesture state. Its filter status shows fist confirmation, release, or neutral-rearm frame progress. Use these values when tuning gesture rules against real webcam poses.

Defaults are centralized in `src/config.py`:

- `CONFIDENCE_THRESHOLD = 0.5`
- `PINCH_THRESHOLD = 0.30`
- `COMMAND_COOLDOWN_SEC = 1.5`
- `CAMERA_ID = 0`

## Adding a gesture

1. Add a state to `GestureState` in `src/recognition/gesture_engine.py`.
2. Add a geometric rule using the shared 21-point layout. Check confidence and guard against invalid reference distances.
3. Add a gesture-to-action mapping to `CommandTranslator.ACTIONS` and a mouse-button mapping when the gesture should click.
4. Extend or add unit tests using synthetic `HandPrediction` data.
5. Update the HUD/documentation when the new gesture needs additional display or configuration.
