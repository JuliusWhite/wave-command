# Recognition and gestures

## Detector backend selection

`HandDetector` in `src/recognition/hand_detector.py` converts each frame into `HandPrediction` objects with a common structure:

- `id`: index in the hand's 21-point layout.
- `x`, `y`: pixel coordinates in the source image.
- `normalized_x`, `normalized_y`: image coordinates scaled to `[0, 1]`.
- `conf`: confidence associated with the point.
- `z`: estimated depth relative to the wrist, in the same pixel scale as `x` (smaller is closer to the camera). MediaPipe provides it; it is `None` for YOLO.
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
| Pinch-ready claw (ring + pinky curled, index + middle relaxed) | Any | `PointerSample` on `GestureResult.pointer` | Cursor movement; pinches click |
| Upright open palm | Vertical | `SCROLL_UP` / `SCROLL_DOWN` | Mouse wheel |
| Index + middle extended, ring + pinky curled | Horizontal swipe | `NEXT_TAB` / `PREVIOUS_TAB` | Ctrl+Tab / Ctrl+Shift+Tab |
| Index only extended | Horizontal swipe | `FORWARD` / `BACK` | Alt+Right / Alt+Left |
| Thumb tip to index tip | Pinch | `PINCH_LEFT_CLICK` | Left-click |
| Thumb tip to middle tip | Pinch | `PINCH_RIGHT_CLICK` | Right-click |

Finger poses are judged by the joint angle at each finger's PIP joint (MCP–PIP–tip): at least `150` degrees counts as extended and at most `130` degrees as curled. Each pose owns one action. An active pointer session takes priority; otherwise the engine checks the open palm, index-only, and two-finger poses in that order. Moving into a different pose resets the movement anchors of the others.

## Keeping gestures apart

Webcam landmarks are noisy, and a hand passes through other poses while changing shape. `GestureEngine` therefore runs each frame through four stages, so one bad frame or a transition is not taken for a different gesture:

1. **Per-frame classification** (`_classify`) judges finger angles and pinch ratios on that frame alone.
2. **Pose stabilizer.** The open-palm, index-only, and two-finger poses feed one *stable pose* (HUD: `Pose:`). With no stable pose, a recognized pose takes over at once. A stable pose survives up to `POSE_RELEASE_FRAMES - 1` (2) mismatched frames, such as a dropped detection or a finger near its angle limit, so a scroll or swipe in progress keeps its anchor. Another pose must be seen on `POSE_SWITCH_FRAMES` (2) consecutive frames to take over, so a single stray frame cannot start a different action.
3. **Routing.** The pointer session or the stable pose owns the hand and drives exactly one action. During gap frames, motion tracking pauses rather than resets.
4. **Pinch filter.** A pinch must be seen on `PINCH_CONFIRM_FRAMES` (2) consecutive frames before it clicks. It then stays *held*, as the same button, until its ratio opens past `PINCH_RELEASE_THRESHOLD` (0.6). The gap above `PINCH_THRESHOLD` (hysteresis) means a pinch that flickers while held clicks once, and a held left pinch cannot turn into a right-click as the thumb slides.

Further rules separate specific pairs:

- **Swipe vs. pointer.** The pointer only arms while no stable pose is active and the palm stays within `POINTER_ARM_MAX_DRIFT_RATIO` (0.35) hand sizes of where arming began. A swipe made with slightly bent fingers is moving, so it never turns into cursor control.
- **Swipe vs. drift.** A swipe must cover its distance within `SWIPE_WINDOW_SEC` (0.8 s), and more sideways than vertically, so slow drift and diagonal moves (such as lowering the hand) never swipe.
- **Swipe vs. return stroke.** After any swipe, motion back the other way is ignored for `SWIPE_RETURN_LOCKOUT_SEC` (1.2 s) or until the pose is released, so returning the hand does not undo the action. Repeating the same swipe works without releasing the pose.
- **Pinch vs. everything else.** Open-palm arming expires `PINCH_ARM_TIMEOUT_SEC` (3 s) after the palm was last seen, and is cleared when a swipe pose takes over, so a stale arming cannot click much later.

## Cursor control

### How to operate it

1. **Start pointing.** Hold the pinch-ready claw: curl the ring and pinky into the palm, and let the index and middle relax with a slight bend, thumb loosely to the side (as if about to pick something up). Keep it, with the hand still, for `POINTER_ENTER_FRAMES` (5) frames, about 0.2 seconds at 30 FPS. Coming from another pose, that pose must end first, which adds about two frames. The HUD shows `CURSOR` with `hold still... 1/5 … 5/5`, then the click hint (in the detailed view, the *Pointer* line counts `arming 1/5 … 5/5` and then shows `moving cursor`).
2. **Move.** The cursor follows your palm centre, not a fingertip. Moving your hand to your right moves the cursor right. Movement is relative, like a mouse: slow movement gives about 400 pixels per hand size of travel for precise targeting, and fast movement up to 1600 for crossing the screen.
3. **Click.** Pinch the thumb to the index tip for a left-click or to the middle tip for a right-click, and hold it for two frames. No open-palm arming is needed while pointing. Open the pinch clearly (ratio above 0.6) before the next click. As soon as either pinch ratio drops below `POINTER_FREEZE_PINCH_RATIO` (0.7), before the click registers, the cursor freezes (HUD: `frozen - finish the pinch to click`), so the click lands where you aimed. Open the pinch to resume movement. The usual 1.5-second click cooldown applies.
4. **Reposition.** While pointing, the cursor stops whenever tracking is lost; bring your hand back to a comfortable spot and carry on. To reach a far edge, move, stop pointing, return your hand, and point again.
5. **Stop pointing.** Open the ring and pinky (for example into an open palm) for `POINTER_EXIT_FRAMES` (3) frames, or take the hand out of view. Detection gaps shorter than that do not end the session.
6. **Other gestures.** While pointing, scrolling and tab/history swipes are off, even if the index and middle straighten. Stop pointing first, then use the open palm, two-finger, or index-only pose. A pose passed through while forming another one, or a claw that is moving, does not start pointing.

**Emergency stop:** move the physical mouse into a screen corner. PyAutoGUI's fail-safe stops the app; hand-driven movement keeps one pixel away from the edges, so it never triggers the fail-safe by itself.

### How it works

- Arming counts only while no other stable pose is active and the palm stays within `POINTER_ARM_MAX_DRIFT_RATIO` hand sizes of where arming began; moving further restarts the count.
- **With depth (MediaPipe)**, the claw is judged in 3D. With the palm facing the camera, a claw's fingers bend *toward the lens*, which looks straight in the flat image, so 2D joint angles read a claw as the two-finger tab pose. The engine therefore measures each finger's 3D flexion: the bend at its PIP joint plus the bend at its DIP joint, from the landmark depths (0 = straight). The MCP joint is left out because finger splay there would read as bend. The claw requires:
  - all four fingers visible;
  - ring and pinky curled: at most 130 degrees in 2D, *or* at least `POINTER_CURLED_FLEX_DEGREES` (100) of 3D flexion;
  - an average index/middle flexion of at least `POINTER_CLAW_MIN_FLEX_DEGREES` (35);
  - neither finger over `POINTER_CLAW_MAX_FLEX_DEGREES` (150, a fist).

  Flexion is smoothed over frames (`POINTER_FLEX_SMOOTHING_ALPHA`, 0.5) because depth estimates jitter. While arming, up to `POINTER_ARM_GRACE_FRAMES` (2) frames without the claw pause the count instead of restarting it; another pose taking over still restarts it.

  The two-finger and index-only poses additionally require their extended fingers to be below `POINTER_CLAW_MIN_FLEX_DEGREES` in 3D, so a claw is never taken for a swipe pose.
- **Without depth (YOLO)**, the 2D rules apply: ring and pinky curled (at most 130 degrees), index and middle bent more than `POINTER_MIN_FINGER_ANGLE_DEGREES` (100, i.e. not a fist), not both straight (that is the two-finger tab pose), and not the index-only pose.
- The HUD's `Flex 3D I/M/R/P` line shows each finger's 3D flexion next to the claw band. If the claw is not recognized, compare the index and middle values with the band; if straight fingers read above 35, raise `POINTER_CLAW_MIN_FLEX_DEGREES`.
- Once started, the session is held by the ring and pinky alone, so pinching or straightening the index cannot end it or trigger a swipe.
- The engine reports a `PointerSample` (palm centre = mean of the wrist and the four finger MCPs, hand size, and a `frozen` flag) on `GestureResult.pointer`.
- `PointerController` in `src/translation/pointer_controller.py` smooths the palm position with a One Euro filter (`POINTER_FILTER_*`): strong smoothing at rest, little lag when moving. It converts the filtered motion to hand sizes, applies speed-dependent gain between `POINTER_SLOW_PX_PER_HAND_SIZE` and `POINTER_FAST_PX_PER_HAND_SIZE`, inverts horizontal motion because the camera image is not mirrored (`POINTER_INVERT_X`), and moves the cursor relatively.
- A frozen or missing sample resets the filter and the anchor, so neither clicking nor repositioning makes the cursor jump.
- The cursor is limited to the primary display.

## Pinch rules

`GestureEngine.evaluate()` requires a hand with 21 points and confidence at least `CONFIDENCE_THRESHOLD` for the wrist (0) and middle-finger MCP (9), which define the hand size. A pinch ratio is only computed when the thumb tip (4) and the corresponding fingertip are also confident. It computes each available finger's fingertip distance from the thumb tip, normalized by hand size:

```text
hand_size = distance(point[0], point[9])
index_pinch_ratio = distance(point[4], point[8]) / hand_size
middle_pinch_ratio = distance(point[4], point[12]) / hand_size
```

Outside a pointer session, each pinch requires the engine to recognize an upright, camera-facing open palm first. The wrist-to-middle-MCP axis must point upward and be close to vertical, and the index-to-pinky MCP span must remain broad enough in the image to indicate a palm facing the camera. The open-hand pose arms pinch input; the next confirmed pinch triggers and consumes that activation, so the user must show the upright open palm again before another pinch. Losing the hand, switching to a swipe pose, or waiting longer than `PINCH_ARM_TIMEOUT_SEC` also resets the gate. Every pinch, armed or in a pointer session, passes the confirmation and hold rules in [Keeping gestures apart](#keeping-gestures-apart). Until armed, pinch poses produce `IDLE`. When an index-tip ratio is at or below `PINCH_THRESHOLD`, the state is `PINCH_LEFT_CLICK`. When a middle-tip ratio is at or below the threshold, the state is `PINCH_RIGHT_CLICK`. If both ratios are within the threshold, the smaller ratio must lead by at least `PINCH_DOMINANCE_MARGIN`; otherwise, the pose is treated as ambiguous and produces `IDLE`. The ratio is based on pixel coordinates, and normalizing by hand size reduces sensitivity to how close the hand is to the camera. A missing hand, a low-confidence required landmark, or an invalid/near-zero reference distance produces `IDLE`.

The close-window gesture and action are temporarily disabled. Their classifier code in `src/recognition/gesture_engine.py` and action mapping/dispatch branch in `src/translation/command_translator.py` are commented out; related tests are skipped so they can be restored later.

## Scrolling

Use the upright, camera-facing open-palm pose and move the wrist vertically. Scrolling starts after the wrist moves `SCROLL_START_DISTANCE_RATIO` hand sizes from its starting position; this keeps small movements from activating scroll while the open hand arms pinches. Once started, the engine measures wrist velocity relative to hand size and elapsed time. It ignores movement below `SCROLL_DEAD_ZONE_RATIO_PER_SEC`, smooths velocity with `SCROLL_SMOOTHING_ALPHA`, caps it at `SCROLL_MAX_HAND_SIZES_PER_SEC`, then converts it to wheel units using `SCROLL_UNITS_PER_HAND_SIZE_PER_SEC`. Fractional wheel values are accumulated and sent as small integer steps, allowing scroll to continue while the hand moves and decay smoothly when it stops. Moving the hand up emits negative wheel units (`SCROLL_DOWN`); moving it down emits positive units (`SCROLL_UP`). Horizontal-dominant movement does not scroll. Scrolling disarms pinch input; release the open-palm pose and present it again before pinching. Scrolling targets the application under the current cursor. PyAutoGUI still sends discrete wheel events, so the result is smoother but not pixel-perfect smooth scrolling.

## Browser tab navigation

Tab switching uses its own pose so it cannot be confused with scrolling: extend the index and middle fingers and curl the ring and pinky (PIP joint angles at least 150 and at most 130 degrees, as for history swipes); thumb pose does not matter. Then swipe the wrist sideways by `TAB_SWIPE_DISTANCE_RATIO` hand sizes within `SWIPE_WINDOW_SEC`. Right-to-left in the camera image emits `PREVIOUS_TAB` (Ctrl+Shift+Tab), and left-to-right emits `NEXT_TAB` (Ctrl+Tab). The camera image is not mirrored, so this follows the same convention as history swipes.

After a switch, bringing the hand back is ignored for `SWIPE_RETURN_LOCKOUT_SEC` or until the pose is released, so the return stroke does not switch back. To switch several tabs in one direction, swipe, return, and swipe again without releasing the pose. The pose is not recognized while either pinch ratio is at or below `PINCH_THRESHOLD`, so a right-click pinch with straight fingers is never taken for a tab swipe. The open palm only scrolls: horizontal open-palm movement does nothing. While the pose is held, pinch actions are suppressed. The translator adds a 1-second cooldown after a tab switch, during which clicks and history navigation are also ignored. The HUD shows whether the two-finger tab pose is currently detected.

## Pinch re-arming after scrolling

When vertical scrolling is detected, pinch input is disarmed even while the hand remains open. Release the open-palm pose, then present a fresh upright, camera-facing open palm before pinching again. This avoids interpreting the end of a scroll as an intentional pinch.

## Browser history navigation

History navigation uses wrist landmark 0 and a swipe-distance ratio. The pose requires all four non-thumb fingers to be visible, with only the index extended (PIP joint angle at least 150 degrees) and middle, ring, and pinky curled (at most 130 degrees); thumb pose does not matter. While this pose is held, pinch actions are suppressed. Swiping right-to-left emits `BACK` (Alt+Left); swiping left-to-right emits `FORWARD` (Alt+Right), each within `SWIPE_WINDOW_SEC`. After a swipe fires, the return stroke is ignored for `SWIPE_RETURN_LOCKOUT_SEC` or until the pose is released, the same as for tab swipes. Ending the index-only pose (beyond the stabilizer's grace frames) resets its movement anchor. The HUD shows whether this pose is currently detected.

## Tuning with the HUD

Press **D** in the video window to switch from the compact HUD (mode and hint only) to the detailed readout. It shows the gesture and raw classifier state, the pointer state (off, arming, moving, or frozen), the stable pose, the pinch gate (armed, held, direct while pointing, or waiting for an upright palm), per-finger curl ratios and 3D flexion (index/middle/ring/pinky), whether the index-only and two-finger poses are detected, and both pinch ratios. Use these values when tuning gesture rules against real webcam poses.

Defaults are centralized in `src/config.py`:

- `CONFIDENCE_THRESHOLD = 0.5`
- `PINCH_THRESHOLD = 0.45`
- `PINCH_DOMINANCE_MARGIN = 0.08`
- `PINCH_CONFIRM_FRAMES = 2`, `PINCH_RELEASE_THRESHOLD = 0.6`, `PINCH_ARM_TIMEOUT_SEC = 3.0`
- `POSE_RELEASE_FRAMES = 3`, `POSE_SWITCH_FRAMES = 2`
- `SCROLL_START_DISTANCE_RATIO = 0.15`
- `SCROLL_DEAD_ZONE_RATIO_PER_SEC = 0.08`
- `SCROLL_SMOOTHING_ALPHA = 0.25`
- `SCROLL_UNITS_PER_HAND_SIZE_PER_SEC = 1500.0`
- `SCROLL_MAX_HAND_SIZES_PER_SEC = 2.5`
- `HISTORY_SWIPE_DISTANCE_RATIO = 1.25`
- `TAB_SWIPE_DISTANCE_RATIO = 1.25`
- `SWIPE_WINDOW_SEC = 0.8`, `SWIPE_RETURN_LOCKOUT_SEC = 1.2`
- `TAB_SWITCH_COOLDOWN_SEC = 1.0` (navigation cooldown)
- `POINTER_ENTER_FRAMES = 5`, `POINTER_EXIT_FRAMES = 3`, `POINTER_ARM_MAX_DRIFT_RATIO = 0.35`
- `POINTER_MIN_FINGER_ANGLE_DEGREES = 100.0` (2D claw rules)
- `POINTER_CLAW_MIN_FLEX_DEGREES = 35.0`, `POINTER_CLAW_MAX_FLEX_DEGREES = 150.0`, `POINTER_CURLED_FLEX_DEGREES = 100.0`, `POINTER_FLEX_SMOOTHING_ALPHA = 0.5` (3D claw rules)
- `POINTER_ARM_GRACE_FRAMES = 2`
- `POINTER_FREEZE_PINCH_RATIO = 0.7`
- `POINTER_FILTER_MIN_CUTOFF_HZ = 1.0`, `POINTER_FILTER_BETA = 0.02`, `POINTER_FILTER_DERIVATIVE_CUTOFF_HZ = 1.0`
- `POINTER_SLOW_PX_PER_HAND_SIZE = 400.0`, `POINTER_FAST_PX_PER_HAND_SIZE = 1600.0`
- `POINTER_ACCEL_START_SPEED = 0.5`, `POINTER_ACCEL_FULL_SPEED = 3.0` (hand sizes per second)
- `POINTER_INVERT_X = True`
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
