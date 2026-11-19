# Wave Command Architecture

## Goals and safety boundary

Wave Command separates hand recognition, gesture interpretation, action mapping, and webcam visualization. Thumb-to-index and thumb-to-middle pinches trigger left and right clicks. A pinch-ready claw pose moves the cursor, and pinches from it click; upright open-palm vertical movements scroll; index-and-middle (two-finger) wrist swipes switch tabs; index-only wrist swipes navigate back/forward in browser history. Each pose owns a single action, so the gestures cannot trigger each other. Close-window gestures are temporarily disabled.

The action mappings are `PINCH_LEFT_CLICK` → `LEFT_CLICK`, `PINCH_RIGHT_CLICK` → `RIGHT_CLICK`, `NEXT_TAB` → Ctrl+Tab, `PREVIOUS_TAB` → Ctrl+Shift+Tab, `BACK` → Alt+Left, and `FORWARD` → Alt+Right. Browser interactions are the first-iteration focus; the project direction remains broader computer interaction.

## Modules

### Recognition service

- `src/recognition/hand_detector.py` owns vision backend loading and frame-to-landmark conversion.
- If `models/hand-pose.pt` exists, it loads the optional Ultralytics YOLO model (installed from `requirements-yolo.txt`). Its pose head must return exactly 21 landmarks for each hand.
- Otherwise, MediaPipe Hands supplies the same 21-point layout. Up to two hands can be detected and drawn.
- Each `Keypoint` contains its integer `id`, pixel coordinates `x` and `y`, confidence `conf`, normalized image coordinates `normalized_x` and `normalized_y`, and depth `z` relative to the wrist (MediaPipe only; `None` for YOLO).
- `src/recognition/gesture_engine.py` evaluates geometry only; it has no camera, model, or OS-control responsibilities.
- An active pointer session (claw pose held still for `POINTER_ENTER_FRAMES`, kept while the ring and pinky stay curled) takes priority. With depth, the claw is recognized by 3D finger flexion, smoothed over frames, because its fingers bend toward the camera and look straight in 2D. The session reports the palm centre as a `PointerSample` on `GestureResult.pointer`, lets pinches click without open-palm arming, freezes the sample while a pinch forms, and disables scrolling and swipes.
- Otherwise, the pose stabilizer picks one stable pose from the upright open palm (scroll), index-only (history), and index-and-middle (tabs) poses. The stable pose drives only its own action and resets the trackers of the others, so the gestures cannot trigger one another.
- History swipes require the index finger extended and the middle, ring, and pinky curled; the thumb pose does not matter. Right-to-left emits `BACK` (Alt+Left), and left-to-right emits `FORWARD` (Alt+Right).
- Scrolling uses the upright open palm's vertical velocity, normalized by hand size and time, and begins only after the wrist moves `SCROLL_START_DISTANCE_RATIO` hand sizes from its starting position. A dead zone filters small movements, an exponential moving average smooths velocity, and fractional wheel deltas are accumulated into small integer steps. Upward hand movement scrolls down; downward movement scrolls up. Horizontal-dominant movement does not scroll.
- Tab swipes require the index and middle fingers extended and the ring and pinky curled, with no active pinch; the thumb pose does not matter. Once the wrist travels `TAB_SWIPE_DISTANCE_RATIO` hand sizes, right-to-left emits `PREVIOUS_TAB` (Ctrl+Shift+Tab) and left-to-right emits `NEXT_TAB` (Ctrl+Tab). The swipe must happen within `SWIPE_WINDOW_SEC`, and the return stroke is ignored for `SWIPE_RETURN_LOCKOUT_SEC` (history swipes follow the same rule).
- A pose stabilizer debounces the open-palm, index-only, and two-finger poses, and pinches are confirmed over frames and held with hysteresis; see [Keeping gestures apart](recognition-and-gestures.md#keeping-gestures-apart).
- Every pinch requires a fresh upright, camera-facing open-hand recognition. The wrist-to-middle-MCP axis must point upward and be close to vertical, and the index-to-pinky MCP span must remain broad in the image. That pose arms pinch input for `PINCH_ARM_TIMEOUT_SEC`, and the next confirmed pinch consumes the activation; another correctly oriented open hand is required before the next pinch. Detected scrolling also disarms pinches; the open-hand pose must be released before it can arm pinches again, preventing a scroll-ending pose from being mistaken for a click. Losing the hand resets the gate as well. While the index-only or two-finger pose is held, pinch states are suppressed. The rules measure pixel-space distances from thumb tip 4 to index tip 8 and middle tip 12, each divided by the wrist-to-middle-MCP distance (0 to 9). The ratio is scale-independent and avoids aspect-ratio distortion from measuring in normalized image coordinates. A ratio at or below `PINCH_THRESHOLD` (0.45) produces the corresponding left- or right-click state. If both ratios are within the threshold, the smaller ratio must lead by at least `PINCH_DOMINANCE_MARGIN` (0.08); otherwise the pose is treated as ambiguous.

### Translation service

`src/translation/command_translator.py` maps pinch states to mouse buttons, scroll states to mouse-wheel input, and swipe states to browser navigation shortcuts. The translator acts only on a transition into a gesture, so holding a pose does not repeatedly trigger an action. Pinch clicks are suppressed until `COMMAND_COOLDOWN_SEC` (1.5 seconds) has elapsed; navigation actions share a 1-second cooldown (`TAB_SWITCH_COOLDOWN_SEC`), and every non-scroll action is also suppressed for that interval after a tab switch. Scroll steps are sent on every frame with a non-zero wheel amount, but the action message is emitted only when scrolling starts or changes direction. Runtime input uses PyAutoGUI with its per-call `PAUSE` set to zero, so dispatching input does not stall the camera loop.

`src/translation/pointer_controller.py` turns `PointerSample` values into cursor motion: a One Euro filter on the palm position, hand-size-normalized relative movement with speed-dependent acceleration, horizontal inversion for the unmirrored camera, and a one-pixel margin from the screen edges so PyAutoGUI's corner fail-safe stays reserved for the physical mouse. Frozen or missing samples reset the anchor, so clicks and repositioning never make the cursor jump.

### Webcam and visualization service

`src/webcam/visualizer.py` owns the OpenCV camera lifecycle, calls a supplied frame-processing function, draws all landmarks and skeleton connections, and renders the HUD. The thumb-to-finger line changes color when either pinch is recognized. By default the HUD is compact: a translucent panel with the current mode in large type (`CURSOR`, `SCROLL`, `TABS`, `HISTORY`, `LEFT CLICK`/`RIGHT CLICK`, `READY`, or `NO HAND`) and a one-line hint, the FPS in the corner, the most recent action as a short banner (for example `NEXT TAB`), and a key reminder. Text has a dark outline so it stays readable on any background. Press **D** to toggle the detailed tuning readout (backend, classifier and filter state, pointer state, stable pose, pinch gate, curl ratios, 3D flexion, swipe-pose detection, and pinch ratios). Press **B** to toggle bypass, **C** to hide the preview (when the overlay is running), and **Q** to exit.

`src/webcam/overlay.py` provides `StatusOverlay`, a borderless, always-on-top Tk window that mirrors the HUD status (`OverlayStatus`: mode, hint, hand in view, last action, camera visibility, bypass). It is driven from the camera loop: the visualizer calls `show()` with the latest status and `pump()` once per frame, so no extra thread is needed. On Windows it is made non-activating and hidden from the taskbar, so clicking it never steals focus from the browser. Its buttons set flags that the visualizer reads once per frame (`consume_camera_toggle()`, `consume_bypass_toggle()`). With the overlay, the camera preview is optional and is skipped entirely while hidden. The overlay uses the brand palette (light electric blue on deep navy); neutral modes (`READY`, `NO HAND`) are shown in brand blues, while mode colors and the orange `BYPASSED` warning match the HUD.

**Bypass.** `Visualizer.bypassed` turns gesture actions off on demand. Detection and recognition keep running so the overlay and HUD stay live, but `main.py` sends no pointer samples and calls `CommandTranslator.observe()` instead of `process()`. `observe()` tracks the current gesture without acting, so a gesture still held when actions resume is not treated as new and cannot fire a stale click or swipe.

### Application entry point

`main.py` wires the services together and opens the status overlay when `SHOW_STATUS_OVERLAY` is set (falling back to the camera preview if Tk is unavailable). For each frame, it requests detections, evaluates the first detected hand (or `IDLE` if none), passes the pointer sample to `PointerController` and the gesture state to the translator (or, while bypassed, resets the pointer and only lets the translator observe the gesture), and returns a `FrameState` to the visualizer. The detector is closed even when the camera loop exits with an error.

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
        |  GestureResult(state, pinch_distance_ratio, scroll_amount, pointer)
        v
PointerController.update(pointer)    -> relative cursor movement
CommandTranslator.process(state, scroll_amount)
        |  mouse click, wheel step, or keyboard shortcut, and action message
        v
Visualizer updates the status overlay and, when shown, draws
landmarks, skeleton, pinch line, state, and alert
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
2. Implement its geometric rule in `GestureEngine._classify` and `GestureEngine.evaluate`, using confidence-checked landmarks and a scale-normalized distance or angle. Prefer a distinct pose over reusing one that already drives an action: sharing the open palm between scrolling and tab swipes made both unreliable.
3. Add the gesture-to-action entry and its mouse-button, wheel, or hotkey dispatch to `CommandTranslator`.
4. Add or update tests for both the geometric rule and transition/cooldown behavior.
5. Update the HUD or architecture documentation if the gesture needs extra visualization or configuration.

Thresholds, camera ID, connections, and model paths are centralized in `src/config.py`.
