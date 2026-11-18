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

## Gesture overview

| Pose | Movement | Gesture state | Action |
|---|---|---|---|
| Upright open palm | Vertical | `SCROLL_UP` / `SCROLL_DOWN` | Mouse wheel |
| Index + middle extended, ring + pinky curled | Horizontal swipe | `NEXT_TAB` / `PREVIOUS_TAB` | Ctrl+Tab / Ctrl+Shift+Tab |
| Index only extended | Horizontal swipe | `FORWARD` / `BACK` | Alt+Right / Alt+Left |
| Thumb tip to index tip | Pinch | `PINCH_LEFT_CLICK` | Left-click |
| Thumb tip to middle tip | Pinch | `PINCH_RIGHT_CLICK` | Right-click |

Finger poses are judged by the joint angle at each finger's PIP joint (MCP–PIP–tip): at least `150` degrees counts as extended and at most `130` degrees as curled. Each pose owns one action, and the engine checks them in the order open palm, index-only, two-finger. Moving into a different pose resets the movement anchors of the others.

## Pinch rules

`GestureEngine.evaluate()` requires a hand with 21 points and confidence at least `CONFIDENCE_THRESHOLD` for the wrist (0) and middle-finger MCP (9), which define the hand size. A pinch ratio is only computed when the thumb tip (4) and the corresponding fingertip are also confident. It computes each available finger's fingertip distance from the thumb tip, normalized by hand size:

```text
hand_size = distance(point[0], point[9])
index_pinch_ratio = distance(point[4], point[8]) / hand_size
middle_pinch_ratio = distance(point[4], point[12]) / hand_size
```

Each pinch requires the engine to recognize an upright, camera-facing open palm first. The wrist-to-middle-MCP axis must point upward and be close to vertical, and the index-to-pinky MCP span must remain broad enough in the image to indicate a palm facing the camera. The open-hand pose arms pinch input; the next recognized pinch triggers and consumes that activation, so the user must show the upright open palm again before another pinch. Losing the hand also resets the gate. Until armed, pinch poses produce `IDLE`. When an index-tip ratio is at or below `PINCH_THRESHOLD`, the state is `PINCH_LEFT_CLICK`. When a middle-tip ratio is at or below the threshold, the state is `PINCH_RIGHT_CLICK`. If both ratios are within the threshold, the smaller ratio must lead by at least `PINCH_DOMINANCE_MARGIN`; otherwise, the pose is treated as ambiguous and produces `IDLE`. The ratio is based on pixel coordinates, and normalizing by hand size reduces sensitivity to how close the hand is to the camera. A missing hand, a low-confidence required landmark, or an invalid/near-zero reference distance produces `IDLE`.

The close-window gesture and action are temporarily disabled. Their classifier code in `src/recognition/gesture_engine.py` and action mapping/dispatch branch in `src/translation/command_translator.py` are commented out; related tests are skipped so they can be restored later.

## Scrolling

Use the upright, camera-facing open-palm pose and move the wrist vertically. Scrolling starts after the wrist moves `SCROLL_START_DISTANCE_RATIO` hand sizes from its starting position; this keeps small movements from activating scroll while the open hand arms pinches. Once started, the engine measures wrist velocity relative to hand size and elapsed time. It ignores movement below `SCROLL_DEAD_ZONE_RATIO_PER_SEC`, smooths velocity with `SCROLL_SMOOTHING_ALPHA`, caps it at `SCROLL_MAX_HAND_SIZES_PER_SEC`, then converts it to wheel units using `SCROLL_UNITS_PER_HAND_SIZE_PER_SEC`. Fractional wheel values are accumulated and sent as small integer steps, allowing scroll to continue while the hand moves and decay smoothly when it stops. Moving the hand up emits negative wheel units (`SCROLL_DOWN`); moving it down emits positive units (`SCROLL_UP`). Horizontal-dominant movement does not scroll. Scrolling disarms pinch input; release the open-palm pose and present it again before pinching. Scrolling targets the application under the current cursor. PyAutoGUI still sends discrete wheel events, so the result is smoother but not pixel-perfect smooth scrolling.

## Browser tab navigation

Tab switching uses its own pose so it cannot be confused with scrolling: extend the index and middle fingers and curl the ring and pinky (PIP joint angles at least 150 and at most 130 degrees, as for history swipes); thumb pose does not matter. Then swipe the wrist sideways by `TAB_SWIPE_DISTANCE_RATIO` hand sizes. Right-to-left in the camera image emits `PREVIOUS_TAB` (Ctrl+Shift+Tab), and left-to-right emits `NEXT_TAB` (Ctrl+Tab). The camera image is not mirrored, so this follows the same convention as history swipes.

After a switch, the pose must be released (any other pose, or the hand leaving the view) before the next one. This keeps the return stroke from switching back. The pose is not recognized while either pinch ratio is at or below `PINCH_THRESHOLD`, so a right-click pinch with straight fingers is never taken for a tab swipe. The open palm only scrolls: horizontal open-palm movement does nothing. While the pose is held, pinch actions are suppressed. The translator adds a 1-second cooldown after a tab switch, during which clicks and history navigation are also ignored. The HUD shows whether the two-finger tab pose is currently detected.

## Pinch re-arming after scrolling

When vertical scrolling is detected, pinch input is disarmed even while the hand remains open. Release the open-palm pose, then present a fresh upright, camera-facing open palm before pinching again. This avoids interpreting the end of a scroll as an intentional pinch.

## Browser history navigation

History navigation uses wrist landmark 0 and a swipe-distance ratio. The pose requires all four non-thumb fingers to be visible, with only the index extended (PIP joint angle at least 150 degrees) and middle, ring, and pinky curled (at most 130 degrees); thumb pose does not matter. While this pose is held, pinch actions are suppressed. Swiping right-to-left emits `BACK` (Alt+Left); swiping left-to-right emits `FORWARD` (Alt+Right). After a swipe fires, the anchor restarts at the current position, so a return stroke of the same length can trigger the opposite action once the 1-second navigation cooldown has passed. Losing the index-only pose resets its movement anchor. The HUD shows whether this pose is currently detected.

## Tuning with the HUD

The HUD shows the gesture and raw classifier state, whether pinches are armed, per-finger curl ratios (index/middle/ring/pinky), whether the index-only and two-finger poses are detected, and both pinch ratios. Use these values when tuning gesture rules against real webcam poses.

Defaults are centralized in `src/config.py`:

- `CONFIDENCE_THRESHOLD = 0.5`
- `PINCH_THRESHOLD = 0.45`
- `PINCH_DOMINANCE_MARGIN = 0.08`
- `SCROLL_START_DISTANCE_RATIO = 0.15`
- `SCROLL_DEAD_ZONE_RATIO_PER_SEC = 0.08`
- `SCROLL_SMOOTHING_ALPHA = 0.25`
- `SCROLL_UNITS_PER_HAND_SIZE_PER_SEC = 1500.0`
- `SCROLL_MAX_HAND_SIZES_PER_SEC = 2.5`
- `HISTORY_SWIPE_DISTANCE_RATIO = 1.25`
- `TAB_SWIPE_DISTANCE_RATIO = 1.25`
- `TAB_SWITCH_COOLDOWN_SEC = 1.0` (navigation cooldown)
- `COMMAND_COOLDOWN_SEC = 1.5`
- `HUD_ALERT_DURATION_SEC = 2.5`
- `CAMERA_ID = 0`

Finger-angle limits (`FINGER_EXTENDED_ANGLE_DEGREES = 150`, `FINGER_CURLED_ANGLE_DEGREES = 130`) and the open-palm shape ratios are defined at the top of `src/recognition/gesture_engine.py`.

## Adding a gesture

1. Add a state to `GestureState` in `src/recognition/gesture_engine.py`.
2. Add a geometric rule using the shared 21-point layout. Check confidence and guard against invalid reference distances. Give the gesture its own pose rather than adding a second movement to a pose that already drives an action.
3. Add a gesture-to-action mapping to `CommandTranslator.ACTIONS` and the appropriate mouse-button or keyboard-shortcut mapping.
4. Extend or add unit tests using synthetic `HandPrediction` data.
5. Update the HUD/documentation when the new gesture needs additional display or configuration.
