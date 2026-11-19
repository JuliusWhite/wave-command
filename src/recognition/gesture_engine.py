"""Geometric gesture rules that operate on detected hand landmarks."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
# from itertools import combinations
from math import acos, degrees, hypot, sqrt

from ..config import (
    CONFIDENCE_THRESHOLD,
    HISTORY_SWIPE_DISTANCE_RATIO,
    PINCH_ARM_TIMEOUT_SEC,
    PINCH_CONFIRM_FRAMES,
    PINCH_DOMINANCE_MARGIN,
    PINCH_RELEASE_THRESHOLD,
    PINCH_THRESHOLD,
    POINTER_ARM_GRACE_FRAMES,
    POINTER_ARM_MAX_DRIFT_RATIO,
    POINTER_CLAW_MAX_FLEX_DEGREES,
    POINTER_CLAW_MIN_FLEX_DEGREES,
    POINTER_CURLED_FLEX_DEGREES,
    POINTER_ENTER_FRAMES,
    POINTER_EXIT_FRAMES,
    POINTER_FLEX_SMOOTHING_ALPHA,
    POINTER_FREEZE_PINCH_RATIO,
    POINTER_MIN_FINGER_ANGLE_DEGREES,
    POSE_RELEASE_FRAMES,
    POSE_SWITCH_FRAMES,
    SCROLL_DEAD_ZONE_RATIO_PER_SEC,
    SCROLL_MAX_HAND_SIZES_PER_SEC,
    SCROLL_SMOOTHING_ALPHA,
    SCROLL_START_DISTANCE_RATIO,
    SCROLL_UNITS_PER_HAND_SIZE_PER_SEC,
    SWIPE_RETURN_LOCKOUT_SEC,
    SWIPE_WINDOW_SEC,
    TAB_SWIPE_DISTANCE_RATIO,
)
from .hand_detector import HandPrediction, Keypoint

FIST_CONFIRM_FRAMES = 3
FIST_RELEASE_FRAMES = 6
FIST_REARM_FRAMES = 3
FINGER_EXTENDED_ANGLE_DEGREES = 150.0
FINGER_CURLED_ANGLE_DEGREES = 130.0
OPEN_HAND_VERTICAL_ALIGNMENT_RATIO = 0.4
OPEN_HAND_PALM_WIDTH_RATIO = 0.55
# FINGERTIP_CLUSTER_THRESHOLD = 0.15


class GestureState(str, Enum):
    IDLE = "IDLE"
    PINCH_LEFT_CLICK = "PINCH_LEFT_CLICK"
    PINCH_RIGHT_CLICK = "PINCH_RIGHT_CLICK"
    SCROLL_UP = "SCROLL_UP"
    SCROLL_DOWN = "SCROLL_DOWN"
    FIST_CLOSE_WINDOW = "FIST_CLOSE_WINDOW"
    NEXT_TAB = "NEXT_TAB"
    PREVIOUS_TAB = "PREVIOUS_TAB"
    BACK = "BACK"
    FORWARD = "FORWARD"


class HandPose(str, Enum):
    """Movement-driving poses tracked by the pose stabilizer."""

    NONE = "NONE"
    OPEN_PALM = "OPEN_PALM"
    INDEX_ONLY = "INDEX_ONLY"
    TWO_FINGER = "TWO_FINGER"


def _bend_3d(first: Keypoint, joint: Keypoint, last: Keypoint) -> float | None:
    """Degrees a joint bends away from straight (0 = straight), in 3D."""
    assert first.z is not None and joint.z is not None and last.z is not None
    ax, ay, az = first.x - joint.x, first.y - joint.y, first.z - joint.z
    bx, by, bz = last.x - joint.x, last.y - joint.y, last.z - joint.z
    lengths = sqrt(ax * ax + ay * ay + az * az) * sqrt(
        bx * bx + by * by + bz * bz
    )
    if lengths <= 1e-6:
        return None
    cosine = (ax * bx + ay * by + az * bz) / lengths
    return 180.0 - degrees(acos(max(-1.0, min(1.0, cosine))))


def _finger_flexion(
    mcp: Keypoint,
    pip: Keypoint,
    dip: Keypoint,
    tip: Keypoint,
) -> float | None:
    """Total 3D bend of a finger at its PIP and DIP joints, in degrees.

    Unlike the 2D joint angle, this sees fingers bending toward the camera.
    The MCP joint is left out because finger splay there reads as bend.
    """
    pip_bend = _bend_3d(mcp, pip, dip)
    dip_bend = _bend_3d(pip, dip, tip)
    if pip_bend is None or dip_bend is None:
        return None
    return pip_bend + dip_bend


PINCH_GESTURES = frozenset(
    {GestureState.PINCH_LEFT_CLICK, GestureState.PINCH_RIGHT_CLICK}
)


@dataclass(frozen=True)
class GestureDiagnostics:
    raw_gesture: GestureState = GestureState.IDLE
    curled_fingers: int = 0
    visible_fingers: int = 0
    curl_ratios: tuple[float | None, ...] = (None, None, None, None)
    # 3D PIP + DIP flexion per finger (index/middle/ring/pinky), when the
    # backend provides depth.
    finger_flex: tuple[float | None, ...] = (None, None, None, None)
    left_pinch_ratio: float | None = None
    right_pinch_ratio: float | None = None
    fist_confirmation_frames: int = 0
    fist_release_frames: int = 0
    fist_rearm_frames: int = 0
    fist_active: bool = False
    fist_rearm_pending: bool = False
    open_hand: bool = False
    pinch_enabled: bool = False
    pinch_held: bool = False
    index_only_hand: bool = False
    two_finger_hand: bool = False
    ring_pinky_curled: bool = False
    stable_pose: HandPose = HandPose.NONE
    pointer_pose: bool = False
    pointer_arming_frames: int = 0
    pointer_active: bool = False
    pointer_frozen: bool = False
    palm_x: float | None = None
    palm_y: float | None = None
    wrist_x: float | None = None
    wrist_y: float | None = None
    hand_size: float | None = None


@dataclass(frozen=True)
class PointerSample:
    """Palm position for cursor control, in camera-image pixels."""

    x: float
    y: float
    hand_size: float
    frozen: bool


@dataclass(frozen=True)
class GestureResult:
    gesture: GestureState
    pinch_distance_ratio: float | None
    scroll_amount: int = 0
    pointer: PointerSample | None = None


class SwipeTracker:
    """Detect quick horizontal wrist swipes, one at a time.

    A swipe fires when the wrist travels ``distance_ratio`` hand sizes
    sideways within ``window_sec``, moving more sideways than vertically, so
    slow drift and diagonal motion never swipe. After a swipe, motion back
    the other way is ignored for ``return_lockout_sec``: returning the hand
    does not undo the action, while repeating the same swipe still works
    without releasing the pose.
    """

    def __init__(
        self,
        distance_ratio: float,
        window_sec: float = SWIPE_WINDOW_SEC,
        return_lockout_sec: float = SWIPE_RETURN_LOCKOUT_SEC,
    ) -> None:
        self.distance_ratio = distance_ratio
        self.window_sec = window_sec
        self.return_lockout_sec = return_lockout_sec
        self._samples: deque[tuple[float, float, float]] = deque()
        self._locked_direction = 0
        self._locked_until = float("-inf")

    def reset(self) -> None:
        self._samples.clear()
        self._locked_direction = 0
        self._locked_until = float("-inf")

    def update(self, now: float, x: float, y: float, hand_size: float) -> int:
        """Add a wrist sample and return the swipe direction, if any.

        ``1`` is a swipe toward smaller image x, ``-1`` toward larger image
        x, and ``0`` means no swipe on this frame.
        """
        self._samples.append((now, x, y))
        while now - self._samples[0][0] > self.window_sec:
            self._samples.popleft()

        direction = 0
        best_distance = 0.0
        for start, sign in (
            (max(self._samples, key=lambda sample: sample[1]), 1),
            (min(self._samples, key=lambda sample: sample[1]), -1),
        ):
            _, start_x, start_y = start
            horizontal = (start_x - x) * sign / hand_size
            vertical = abs(start_y - y) / hand_size
            if (
                horizontal >= self.distance_ratio
                and horizontal > vertical
                and horizontal > best_distance
            ):
                direction, best_distance = sign, horizontal
        if direction == 0:
            return 0

        # This stroke is used up whether it fires or not; the next swipe
        # starts from here.
        self._samples.clear()
        self._samples.append((now, x, y))
        if (
            direction == -self._locked_direction
            and now < self._locked_until
        ):
            return 0
        self._locked_direction = direction
        self._locked_until = now + self.return_lockout_sec
        return direction


class GestureEngine:
    """Evaluate geometric rules independently from model inference.

    Each frame runs through four stages:

    1. ``_classify`` judges finger poses and pinch ratios on that frame alone.
    2. The pose stabilizer turns the per-frame open-palm, index-only, and
       two-finger poses into one stable pose that survives brief detection
       glitches and resists takeover by stray frames of another pose.
    3. The stable pose (or an active pointer session) owns the hand and
       routes motion to exactly one action: scroll, history, tabs, or cursor.
    4. Pinches are confirmed over several frames and held with hysteresis,
       so each deliberate pinch clicks once.
    """

    def __init__(
        self,
        pinch_threshold: float = PINCH_THRESHOLD,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.pinch_threshold = pinch_threshold
        self._clock = clock
        self._stable_gesture = GestureState.IDLE
        self._fist_active = False
        self._fist_candidate_frames = 0
        self._fist_release_frames = 0
        self._fist_rearm_pending = False
        self._fist_rearm_frames = 0
        self._stable_pose = HandPose.NONE
        self._pose_mismatch_frames = 0
        self._pose_candidate = HandPose.NONE
        self._pose_candidate_frames = 0
        self._pinch_enabled = False
        self._pinch_armed_at = float("-inf")
        self._pinch_waiting_for_open_hand_release = False
        self._pinch_candidate: GestureState | None = None
        self._pinch_candidate_frames = 0
        self._pinch_held: GestureState | None = None
        self._scroll_last_x: float | None = None
        self._scroll_last_y: float | None = None
        self._scroll_last_size: float | None = None
        self._scroll_last_time: float | None = None
        self._scroll_origin_x: float | None = None
        self._scroll_origin_y: float | None = None
        self._scroll_origin_size: float | None = None
        self._scroll_started = False
        self._scroll_velocity = 0.0
        self._scroll_remainder = 0.0
        self._history_swipe = SwipeTracker(HISTORY_SWIPE_DISTANCE_RATIO)
        self._tab_swipe = SwipeTracker(TAB_SWIPE_DISTANCE_RATIO)
        self._pointer_active = False
        self._pointer_arming_frames = 0
        self._pointer_arm_origin: tuple[float, float] | None = None
        self._pointer_arm_miss_frames = 0
        # Smoothed 3D flexion per finger; depth estimates jitter per frame.
        self._smoothed_flex: list[float | None] = [None, None, None, None]
        self._pointer_exit_frames = 0
        self._pointer_sample: PointerSample | None = None
        self._diagnostics = GestureDiagnostics()

    @property
    def diagnostics(self) -> GestureDiagnostics:
        """Return classifier features and current gesture-filter progress."""
        return replace(
            self._diagnostics,
            fist_confirmation_frames=self._fist_candidate_frames,
            fist_release_frames=self._fist_release_frames,
            fist_rearm_frames=self._fist_rearm_frames,
            fist_active=self._fist_active,
            fist_rearm_pending=self._fist_rearm_pending,
        )

    def evaluate(self, hand: HandPrediction | None) -> GestureResult:
        """Classify pointing, pinches, scrolling, and two swipe gestures.

        Each pose owns one action: the pinch-ready pointer pose moves the
        cursor, the open palm scrolls, an index-only hand navigates history,
        and an index-and-middle hand switches tabs.
        """
        result = self._evaluate(hand)
        return replace(result, pointer=self._pointer_sample)

    def _evaluate(self, hand: HandPrediction | None) -> GestureResult:
        now = self._clock()
        if hand is None:
            self._smoothed_flex = [None, None, None, None]
        raw, diagnostics = self._classify(hand)
        raw_pinch = raw.gesture if raw.gesture in PINCH_GESTURES else None
        self._pointer_sample = None
        pointer_frozen = False

        pose = self._update_pose(self._pose_label(diagnostics))
        pointer_session = self._update_pointer_session(diagnostics, pose)

        if pointer_session:
            result = self._evaluate_pointer(raw, raw_pinch, diagnostics)
            pointer_frozen = (
                self._pointer_sample is not None
                and self._pointer_sample.frozen
            )
            pose = HandPose.NONE
        else:
            result = self._evaluate_poses(
                raw, raw_pinch, diagnostics, hand, pose, now
            )

        diagnostics = replace(
            diagnostics,
            raw_gesture=(
                result.gesture
                if result.gesture != GestureState.IDLE
                else raw.gesture
            ),
            pinch_enabled=self._pinch_enabled,
            pinch_held=self._pinch_held is not None,
            stable_pose=pose,
            pointer_arming_frames=self._pointer_arming_frames,
            pointer_active=pointer_session,
            pointer_frozen=pointer_frozen,
        )
        self._diagnostics = diagnostics
        return self._filter_fist(result)

    def _evaluate_pointer(
        self,
        raw: GestureResult,
        raw_pinch: GestureState | None,
        diagnostics: GestureDiagnostics,
    ) -> GestureResult:
        """Pointing owns the hand: no scrolling or swipes, pinches click.

        Pinches click directly, without the open-palm arming used elsewhere.
        """
        self._set_pose(HandPose.NONE)
        tracking = (
            diagnostics.ring_pinky_curled
            and diagnostics.palm_x is not None
            and diagnostics.palm_y is not None
            and diagnostics.hand_size is not None
        )
        pinch = self._filter_pinch(raw_pinch, diagnostics, allowed=tracking)
        if tracking:
            assert diagnostics.palm_x is not None
            assert diagnostics.palm_y is not None
            assert diagnostics.hand_size is not None
            pinch_ratios = [
                ratio
                for ratio in (
                    diagnostics.left_pinch_ratio,
                    diagnostics.right_pinch_ratio,
                )
                if ratio is not None
            ]
            # Freeze while a pinch is forming, and while it is held, so the
            # click lands where the user was pointing.
            frozen = pinch is not None or (
                bool(pinch_ratios)
                and min(pinch_ratios) < POINTER_FREEZE_PINCH_RATIO
            )
            self._pointer_sample = PointerSample(
                diagnostics.palm_x,
                diagnostics.palm_y,
                diagnostics.hand_size,
                frozen,
            )
        if pinch is None:
            return GestureResult(GestureState.IDLE, raw.pinch_distance_ratio)
        return GestureResult(pinch, raw.pinch_distance_ratio)

    def _evaluate_poses(
        self,
        raw: GestureResult,
        raw_pinch: GestureState | None,
        diagnostics: GestureDiagnostics,
        hand: HandPrediction | None,
        pose: HandPose,
        now: float,
    ) -> GestureResult:
        """Route a non-pointing frame to the action of its stable pose."""
        self._update_pinch_arming(hand, diagnostics, pose, now)
        # Swipe poses never pinch: a thumb drifting toward a finger while
        # swiping must not click.
        pinch = self._filter_pinch(
            raw_pinch,
            diagnostics,
            allowed=(
                self._pinch_enabled
                and pose in {HandPose.NONE, HandPose.OPEN_PALM}
            ),
        )
        if pinch is not None:
            # A held pinch owns the hand; no motion is tracked meanwhile.
            self._reset_movement_tracking()
            return GestureResult(pinch, raw.pinch_distance_ratio)

        idle = GestureResult(GestureState.IDLE, raw.pinch_distance_ratio)
        wrist_x = diagnostics.wrist_x
        wrist_y = diagnostics.wrist_y
        hand_size = diagnostics.hand_size
        if wrist_x is None or wrist_y is None or hand_size is None:
            # No usable hand this frame: keep tracking paused while the
            # stable pose rides out the gap.
            return idle

        if pose == HandPose.OPEN_PALM:
            if raw_pinch is not None:
                # A forming pinch is not scroll motion.
                self._reset_scroll_tracking()
                return idle
            if not diagnostics.open_hand:
                return idle
            scroll_amount = self._scroll_from_open_hand(
                wrist_x, wrist_y, hand_size
            )
            if not scroll_amount:
                return idle
            # Scrolling consumes the pinch activation; the open palm must
            # be released and shown again before the next pinch.
            self._pinch_enabled = False
            self._pinch_waiting_for_open_hand_release = True
            scroll_gesture = (
                GestureState.SCROLL_DOWN
                if scroll_amount < 0
                else GestureState.SCROLL_UP
            )
            return GestureResult(scroll_gesture, None, scroll_amount)

        if pose == HandPose.INDEX_ONLY:
            if not diagnostics.index_only_hand:
                return GestureResult(GestureState.IDLE, None)
            direction = self._history_swipe.update(
                now, wrist_x, wrist_y, hand_size
            )
            gesture = {
                1: GestureState.BACK,
                -1: GestureState.FORWARD,
            }.get(direction, GestureState.IDLE)
            return GestureResult(gesture, None)

        if pose == HandPose.TWO_FINGER:
            if not diagnostics.two_finger_hand:
                return GestureResult(GestureState.IDLE, None)
            direction = self._tab_swipe.update(
                now, wrist_x, wrist_y, hand_size
            )
            gesture = {
                1: GestureState.PREVIOUS_TAB,
                -1: GestureState.NEXT_TAB,
            }.get(direction, GestureState.IDLE)
            return GestureResult(gesture, None)

        return idle

    def _filter_fist(self, result: GestureResult) -> GestureResult:
        """Confirm, hold, and rearm the (currently disabled) fist gesture."""
        if self._fist_rearm_pending:
            if result.gesture == GestureState.IDLE:
                self._fist_rearm_frames += 1
                if self._fist_rearm_frames >= FIST_REARM_FRAMES:
                    self._fist_rearm_pending = False
                    self._stable_gesture = GestureState.IDLE
                return GestureResult(GestureState.IDLE, result.pinch_distance_ratio)

            self._fist_rearm_frames = 0
            return GestureResult(GestureState.IDLE, None)

        if self._fist_active:
            if result.gesture == GestureState.FIST_CLOSE_WINDOW:
                self._fist_release_frames = 0
                return GestureResult(GestureState.FIST_CLOSE_WINDOW, None)

            self._fist_release_frames += 1
            if self._fist_release_frames < FIST_RELEASE_FRAMES:
                return GestureResult(GestureState.FIST_CLOSE_WINDOW, None)

            self._fist_active = False
            self._fist_release_frames = 0
            self._stable_gesture = GestureState.IDLE
            self._fist_rearm_pending = True
            self._fist_rearm_frames = 0
            return GestureResult(GestureState.IDLE, None)

        if result.gesture == GestureState.FIST_CLOSE_WINDOW:
            self._fist_candidate_frames += 1
            if self._fist_candidate_frames < FIST_CONFIRM_FRAMES:
                return GestureResult(
                    self._stable_gesture,
                    result.pinch_distance_ratio,
                )

            self._fist_active = True
            self._fist_candidate_frames = 0
            self._stable_gesture = GestureState.FIST_CLOSE_WINDOW
            return result

        self._fist_candidate_frames = 0
        self._stable_gesture = result.gesture
        return result

    @staticmethod
    def _pose_label(diagnostics: GestureDiagnostics) -> HandPose:
        if diagnostics.open_hand:
            return HandPose.OPEN_PALM
        if diagnostics.index_only_hand:
            return HandPose.INDEX_ONLY
        if diagnostics.two_finger_hand:
            return HandPose.TWO_FINGER
        return HandPose.NONE

    def _update_pose(self, label: HandPose) -> HandPose:
        """Debounce the per-frame pose into a stable pose.

        With no stable pose, a recognized pose takes over at once. A stable
        pose survives up to ``POSE_RELEASE_FRAMES - 1`` mismatched frames
        (a glitch, a missing hand, a finger near its angle limit), so the
        scroll or swipe in progress keeps its anchor. A different pose must
        be seen on ``POSE_SWITCH_FRAMES`` consecutive frames to take over.
        """
        if label == self._stable_pose:
            self._pose_mismatch_frames = 0
            self._pose_candidate = HandPose.NONE
            self._pose_candidate_frames = 0
            return self._stable_pose

        if label == HandPose.NONE:
            self._pose_candidate = HandPose.NONE
            self._pose_candidate_frames = 0
        elif label == self._pose_candidate:
            self._pose_candidate_frames += 1
        else:
            self._pose_candidate = label
            self._pose_candidate_frames = 1
        self._pose_mismatch_frames += 1

        if (
            self._stable_pose == HandPose.NONE
            or self._pose_candidate_frames >= POSE_SWITCH_FRAMES
            or self._pose_mismatch_frames >= POSE_RELEASE_FRAMES
        ):
            self._set_pose(label)
        return self._stable_pose

    def _set_pose(self, pose: HandPose) -> None:
        if pose != self._stable_pose:
            # A new pose starts its motion from scratch.
            self._reset_movement_tracking()
        self._stable_pose = pose
        self._pose_mismatch_frames = 0
        self._pose_candidate = HandPose.NONE
        self._pose_candidate_frames = 0

    def _update_pinch_arming(
        self,
        hand: HandPrediction | None,
        diagnostics: GestureDiagnostics,
        pose: HandPose,
        now: float,
    ) -> None:
        """Arm one pinch from an upright open palm; disarm when stale.

        Arming lapses when the hand is lost, when a swipe pose takes over,
        or ``PINCH_ARM_TIMEOUT_SEC`` after the palm was last seen, so an old
        open palm cannot cause a click much later.
        """
        if self._pinch_waiting_for_open_hand_release:
            if not diagnostics.open_hand:
                self._pinch_waiting_for_open_hand_release = False
        elif diagnostics.open_hand:
            self._pinch_enabled = True
            self._pinch_armed_at = now
        if (
            hand is None
            or pose in {HandPose.INDEX_ONLY, HandPose.TWO_FINGER}
            or now - self._pinch_armed_at > PINCH_ARM_TIMEOUT_SEC
        ):
            self._pinch_enabled = False

    def _filter_pinch(
        self,
        raw_pinch: GestureState | None,
        diagnostics: GestureDiagnostics,
        allowed: bool,
    ) -> GestureState | None:
        """Confirm a pinch over frames, then hold it until clearly released.

        A pinch must be seen ``PINCH_CONFIRM_FRAMES`` times in a row before
        it clicks. It then stays held, as the same button, until its ratio
        opens past ``PINCH_RELEASE_THRESHOLD`` or its landmarks are lost.
        The translator acts on the start of a held pinch, so one physical
        pinch is one click.
        """
        if self._pinch_held is not None:
            held_ratio = (
                diagnostics.left_pinch_ratio
                if self._pinch_held == GestureState.PINCH_LEFT_CLICK
                else diagnostics.right_pinch_ratio
            )
            if (
                held_ratio is not None
                and held_ratio <= PINCH_RELEASE_THRESHOLD
            ):
                return self._pinch_held
            self._pinch_held = None

        if raw_pinch is None or not allowed:
            self._pinch_candidate = None
            self._pinch_candidate_frames = 0
            return None
        if raw_pinch == self._pinch_candidate:
            self._pinch_candidate_frames += 1
        else:
            self._pinch_candidate = raw_pinch
            self._pinch_candidate_frames = 1
        if self._pinch_candidate_frames < PINCH_CONFIRM_FRAMES:
            return None

        self._pinch_held = raw_pinch
        self._pinch_candidate = None
        self._pinch_candidate_frames = 0
        # The click consumes the open-palm arming.
        self._pinch_enabled = False
        return raw_pinch

    def _update_pointer_session(
        self,
        diagnostics: GestureDiagnostics,
        pose: HandPose,
    ) -> bool:
        """Start, keep, or end the pointer session for this frame.

        The pointer pose must be held, with the palm roughly still and no
        other stable pose active, for ``POINTER_ENTER_FRAMES``. Passing
        through the pose while forming another one, or swiping with a bent
        finger, therefore does not start pointing. Once started, the session
        lasts while the ring and pinky stay curled, so pinching or
        straightening the index finger does not end it.
        ``POINTER_EXIT_FRAMES`` tolerates brief detection loss.
        """
        if self._pointer_active:
            if diagnostics.ring_pinky_curled:
                self._pointer_exit_frames = 0
                return True
            self._pointer_exit_frames += 1
            if self._pointer_exit_frames < POINTER_EXIT_FRAMES:
                return True
            self._pointer_active = False
            self._pointer_exit_frames = 0
            self._pointer_arming_frames = 0
            return False

        if (
            diagnostics.pointer_pose
            and pose == HandPose.NONE
            and diagnostics.palm_x is not None
            and diagnostics.palm_y is not None
            and diagnostics.hand_size is not None
        ):
            origin = self._pointer_arm_origin
            if origin is None or hypot(
                diagnostics.palm_x - origin[0],
                diagnostics.palm_y - origin[1],
            ) > POINTER_ARM_MAX_DRIFT_RATIO * diagnostics.hand_size:
                # Moving too much: restart arming from here.
                self._pointer_arm_origin = (
                    diagnostics.palm_x,
                    diagnostics.palm_y,
                )
                self._pointer_arming_frames = 1
            else:
                self._pointer_arming_frames += 1
            self._pointer_arm_miss_frames = 0
            if self._pointer_arming_frames >= POINTER_ENTER_FRAMES:
                self._pointer_active = True
                self._pointer_arming_frames = 0
                self._pointer_arm_origin = None
                self._pointer_exit_frames = 0
                return True
        elif (
            pose == HandPose.NONE
            and self._pointer_arming_frames
            and self._pointer_arm_miss_frames < POINTER_ARM_GRACE_FRAMES
        ):
            # A brief miss (noise, a dropped detection) pauses arming
            # instead of restarting it; another pose taking over still does.
            self._pointer_arm_miss_frames += 1
        else:
            self._pointer_arming_frames = 0
            self._pointer_arm_origin = None
            self._pointer_arm_miss_frames = 0
        return False

    def _scroll_from_open_hand(
        self,
        wrist_x: float,
        wrist_y: float,
        hand_size: float,
    ) -> int:
        now = self._clock()
        if (
            self._scroll_last_x is None
            or self._scroll_last_y is None
            or self._scroll_last_size is None
            or self._scroll_last_time is None
            or self._scroll_origin_x is None
            or self._scroll_origin_y is None
            or self._scroll_origin_size is None
        ):
            self._scroll_origin_x = wrist_x
            self._scroll_origin_y = wrist_y
            self._scroll_origin_size = hand_size
            self._scroll_started = False
            self._update_scroll_position(wrist_x, wrist_y, hand_size, now)
            return 0

        if not self._scroll_started:
            origin_size = (self._scroll_origin_size + hand_size) / 2
            vertical_displacement_ratio = (
                self._scroll_origin_y - wrist_y
            ) / origin_size
            horizontal_displacement_ratio = (
                self._scroll_origin_x - wrist_x
            ) / origin_size
            if (
                abs(vertical_displacement_ratio)
                < SCROLL_START_DISTANCE_RATIO
                or abs(vertical_displacement_ratio)
                < abs(horizontal_displacement_ratio)
            ):
                if abs(horizontal_displacement_ratio) > abs(
                    vertical_displacement_ratio
                ):
                    self._scroll_origin_x = wrist_x
                    self._scroll_origin_y = wrist_y
                    self._scroll_origin_size = hand_size
                self._update_scroll_position(wrist_x, wrist_y, hand_size, now)
                return 0
            self._scroll_started = True

        return self._scroll_step(wrist_x, wrist_y, hand_size, now)

    def _scroll_step(
        self,
        wrist_x: float,
        wrist_y: float,
        hand_size: float,
        now: float,
    ) -> int:
        """Convert locked vertical palm velocity into integer wheel steps."""
        assert self._scroll_last_x is not None
        assert self._scroll_last_y is not None
        assert self._scroll_last_size is not None
        assert self._scroll_last_time is not None
        elapsed = now - self._scroll_last_time
        if elapsed <= 0:
            # A coarse clock can repeat a timestamp; keep the previous sample
            # so the next frame measures velocity over the combined interval.
            return 0
        reference_size = (self._scroll_last_size + hand_size) / 2

        horizontal_velocity = (
            (self._scroll_last_x - wrist_x) / reference_size / elapsed
        )
        vertical_velocity = (
            (self._scroll_last_y - wrist_y) / reference_size / elapsed
        )
        vertical_motion = (
            abs(vertical_velocity) > abs(horizontal_velocity)
            and abs(vertical_velocity) >= SCROLL_DEAD_ZONE_RATIO_PER_SEC
        )
        target_velocity = 0.0
        if vertical_motion:
            target_velocity = max(
                -SCROLL_MAX_HAND_SIZES_PER_SEC,
                min(SCROLL_MAX_HAND_SIZES_PER_SEC, vertical_velocity),
            )

        self._scroll_velocity = (
            SCROLL_SMOOTHING_ALPHA * target_velocity
            + (1.0 - SCROLL_SMOOTHING_ALPHA) * self._scroll_velocity
        )
        if abs(self._scroll_velocity) < SCROLL_DEAD_ZONE_RATIO_PER_SEC:
            self._scroll_velocity = 0.0
            self._scroll_remainder = 0.0
        else:
            integration_time = min(elapsed, 0.1)
            self._scroll_remainder += (
                -self._scroll_velocity
                * SCROLL_UNITS_PER_HAND_SIZE_PER_SEC
                * integration_time
            )

        amount = int(self._scroll_remainder)
        self._scroll_remainder -= amount
        self._update_scroll_position(wrist_x, wrist_y, hand_size, now)
        return amount

    def _update_scroll_position(
        self,
        wrist_x: float,
        wrist_y: float,
        hand_size: float,
        timestamp: float,
    ) -> None:
        self._scroll_last_x = wrist_x
        self._scroll_last_y = wrist_y
        self._scroll_last_size = hand_size
        self._scroll_last_time = timestamp

    def _reset_scroll_tracking(self) -> None:
        self._scroll_last_x = None
        self._scroll_last_y = None
        self._scroll_last_size = None
        self._scroll_last_time = None
        self._scroll_origin_x = None
        self._scroll_origin_y = None
        self._scroll_origin_size = None
        self._scroll_started = False
        self._scroll_velocity = 0.0
        self._scroll_remainder = 0.0

    def _reset_movement_tracking(self) -> None:
        self._reset_scroll_tracking()
        self._history_swipe.reset()
        self._tab_swipe.reset()

    def _smooth_flex(
        self,
        finger_flex: list[float | None],
    ) -> list[float | None]:
        """Low-pass each finger's 3D flexion; depth estimates are noisy.

        A finger without a measurement this frame restarts its average.
        """
        for index, flex in enumerate(finger_flex):
            previous = self._smoothed_flex[index]
            if flex is None or previous is None:
                self._smoothed_flex[index] = flex
            else:
                self._smoothed_flex[index] = previous + (
                    POINTER_FLEX_SMOOTHING_ALPHA * (flex - previous)
                )
        return list(self._smoothed_flex)

    def _classify(
        self,
        hand: HandPrediction | None,
    ) -> tuple[GestureResult, GestureDiagnostics]:
        """Classify finger pose and pinches normalized by hand size."""
        if hand is None or len(hand.keypoints) != 21:
            return GestureResult(GestureState.IDLE, None), GestureDiagnostics()

        wrist = hand.keypoints[0]
        thumb_tip = hand.keypoints[4]
        index_tip = hand.keypoints[8]
        middle_mcp = hand.keypoints[9]
        middle_tip = hand.keypoints[12]
        finger_mcps = (5, 9, 13, 17)
        finger_pips = (6, 10, 14, 18)
        finger_tips = (8, 12, 16, 20)
        # Temporarily disabled close gesture; retain this code for reactivation.
        # all_fingertips = (thumb_tip,) + tuple(
        #     hand.keypoints[index] for index in finger_tips
        # )

        if min(
            wrist.conf,
            middle_mcp.conf,
        ) < CONFIDENCE_THRESHOLD:
            return GestureResult(GestureState.IDLE, None), GestureDiagnostics()

        hand_size = hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y)
        if hand_size <= 1e-6:
            return GestureResult(GestureState.IDLE, None), GestureDiagnostics()

        curled_fingers = 0
        visible_fingers = 0
        curled_by_tip: dict[int, bool] = {}
        curl_ratios: list[float | None] = [None, None, None, None]
        finger_angles: list[float | None] = [None, None, None, None]
        # 3D flexion (degrees) needs depth on every landmark.
        has_depth = all(point.z is not None for point in hand.keypoints)
        finger_flex: list[float | None] = [None, None, None, None]
        for finger_index, mcp_index, pip_index, tip_index in zip(
            range(4), finger_mcps, finger_pips, finger_tips
        ):
            mcp = hand.keypoints[mcp_index]
            pip = hand.keypoints[pip_index]
            tip = hand.keypoints[tip_index]
            if min(mcp.conf, pip.conf, tip.conf) < CONFIDENCE_THRESHOLD:
                continue
            visible_fingers += 1
            tip_distance = hypot(tip.x - mcp.x, tip.y - mcp.y)
            pip_distance = hypot(pip.x - mcp.x, pip.y - mcp.y)
            curl_ratio = (
                tip_distance / pip_distance
                if pip_distance > 1e-6
                else None
            )
            curl_ratios[finger_index] = curl_ratio
            tip_segment_length = hypot(tip.x - pip.x, tip.y - pip.y)
            if pip_distance > 1e-6 and tip_segment_length > 1e-6:
                cosine = (
                    (mcp.x - pip.x) * (tip.x - pip.x)
                    + (mcp.y - pip.y) * (tip.y - pip.y)
                ) / (pip_distance * tip_segment_length)
                finger_angles[finger_index] = degrees(
                    acos(max(-1.0, min(1.0, cosine)))
                )
            if has_depth:
                finger_flex[finger_index] = _finger_flexion(
                    mcp, pip, hand.keypoints[pip_index + 1], tip
                )
            is_curled = curl_ratio is not None and curl_ratio < 1.0
            curled_by_tip[tip_index] = is_curled
            if is_curled:
                curled_fingers += 1

        left_pinch_ratio = None
        if (
            thumb_tip.conf >= CONFIDENCE_THRESHOLD
            and index_tip.conf >= CONFIDENCE_THRESHOLD
        ):
            index_distance = hypot(
                index_tip.x - thumb_tip.x,
                index_tip.y - thumb_tip.y,
            )
            left_pinch_ratio = index_distance / hand_size

        right_pinch_ratio = None
        if (
            thumb_tip.conf >= CONFIDENCE_THRESHOLD
            and middle_tip.conf >= CONFIDENCE_THRESHOLD
        ):
            middle_distance = hypot(
                middle_tip.x - thumb_tip.x,
                middle_tip.y - thumb_tip.y,
            )
            right_pinch_ratio = middle_distance / hand_size

        open_hand = (
            thumb_tip.conf >= CONFIDENCE_THRESHOLD
            and visible_fingers == len(finger_tips)
            and middle_mcp.y < wrist.y
            and abs(middle_mcp.x - wrist.x)
            <= hand_size * OPEN_HAND_VERTICAL_ALIGNMENT_RATIO
            and hypot(
                hand.keypoints[17].x - hand.keypoints[5].x,
                hand.keypoints[17].y - hand.keypoints[5].y,
            ) >= hand_size * OPEN_HAND_PALM_WIDTH_RATIO
            and all(
                angle is not None
                and angle >= FINGER_EXTENDED_ANGLE_DEGREES
                for angle in finger_angles
            )
            and left_pinch_ratio is not None
            and left_pinch_ratio > self.pinch_threshold
            and right_pinch_ratio is not None
            and right_pinch_ratio > self.pinch_threshold
        )
        finger_flex = self._smooth_flex(finger_flex)
        # Average index/middle 3D flexion. With depth, it separates the claw
        # (bent toward the camera) from the straight two-finger and
        # index-only poses, which look alike in the flat image.
        claw_flex = (
            (finger_flex[0] + finger_flex[1]) / 2
            if finger_flex[0] is not None and finger_flex[1] is not None
            else None
        )
        fingers_look_bent_in_3d = (
            claw_flex is not None
            and claw_flex >= POINTER_CLAW_MIN_FLEX_DEGREES
        )
        # Only the index's own bend matters here: the middle is folded.
        index_looks_bent_in_3d = (
            finger_flex[0] is not None
            and finger_flex[0] >= POINTER_CLAW_MIN_FLEX_DEGREES
        )
        index_only_hand = (
            visible_fingers == len(finger_tips)
            and not index_looks_bent_in_3d
            and finger_angles[0] is not None
            and finger_angles[0] >= FINGER_EXTENDED_ANGLE_DEGREES
            and finger_angles[1] is not None
            and finger_angles[1] <= FINGER_CURLED_ANGLE_DEGREES
            and finger_angles[2] is not None
            and finger_angles[2] <= FINGER_CURLED_ANGLE_DEGREES
            and finger_angles[3] is not None
            and finger_angles[3] <= FINGER_CURLED_ANGLE_DEGREES
        )
        # Index and middle extended, ring and pinky curled. An active pinch
        # is excluded so a right-click pinch is never taken for this pose.
        two_finger_hand = (
            visible_fingers == len(finger_tips)
            and not fingers_look_bent_in_3d
            and finger_angles[0] is not None
            and finger_angles[0] >= FINGER_EXTENDED_ANGLE_DEGREES
            and finger_angles[1] is not None
            and finger_angles[1] >= FINGER_EXTENDED_ANGLE_DEGREES
            and finger_angles[2] is not None
            and finger_angles[2] <= FINGER_CURLED_ANGLE_DEGREES
            and finger_angles[3] is not None
            and finger_angles[3] <= FINGER_CURLED_ANGLE_DEGREES
            and (
                left_pinch_ratio is None
                or left_pinch_ratio > self.pinch_threshold
            )
            and (
                right_pinch_ratio is None
                or right_pinch_ratio > self.pinch_threshold
            )
        )

        def curled_for_pointing(finger_index: int) -> bool:
            # A finger folded toward the camera can look straight in 2D, so
            # with depth a 3D fold also counts.
            angle = finger_angles[finger_index]
            flex = finger_flex[finger_index]
            return (
                angle is not None and angle <= FINGER_CURLED_ANGLE_DEGREES
            ) or (flex is not None and flex >= POINTER_CURLED_FLEX_DEGREES)

        ring_pinky_curled = (
            visible_fingers == len(finger_tips)
            and curled_for_pointing(2)
            and curled_for_pointing(3)
        )
        # Pinch-ready pose: ring and pinky curled, index and middle relaxed
        # (bent, but not curled into a fist and not both straight, which is
        # the two-finger tab pose).
        if has_depth:
            pointer_pose = (
                ring_pinky_curled
                and fingers_look_bent_in_3d
                and finger_flex[0] is not None
                and finger_flex[1] is not None
                and max(finger_flex[0], finger_flex[1])
                <= POINTER_CLAW_MAX_FLEX_DEGREES
                and not index_only_hand
            )
        else:
            pointer_pose = (
                ring_pinky_curled
                and finger_angles[0] is not None
                and finger_angles[1] is not None
                and finger_angles[0] > POINTER_MIN_FINGER_ANGLE_DEGREES
                and finger_angles[1] > POINTER_MIN_FINGER_ANGLE_DEGREES
                and not (
                    finger_angles[0] >= FINGER_EXTENDED_ANGLE_DEGREES
                    and finger_angles[1] >= FINGER_EXTENDED_ANGLE_DEGREES
                )
                and not index_only_hand
            )
        palm_x = palm_y = None
        if visible_fingers == len(finger_tips):
            palm_points = [wrist] + [
                hand.keypoints[index] for index in finger_mcps
            ]
            palm_x = sum(point.x for point in palm_points) / len(palm_points)
            palm_y = sum(point.y for point in palm_points) / len(palm_points)

        pinch_ratios = []
        if (
            left_pinch_ratio is not None
            and not curled_by_tip.get(8, False)
        ):
            pinch_ratios.append(
                (GestureState.PINCH_LEFT_CLICK, left_pinch_ratio)
            )
        if (
            right_pinch_ratio is not None
            and not curled_by_tip.get(12, False)
        ):
            pinch_ratios.append(
                (GestureState.PINCH_RIGHT_CLICK, right_pinch_ratio)
            )

        # Temporarily disabled close gesture; uncomment to restore it.
        # fingertips_together = all(
        #     tip.conf >= CONFIDENCE_THRESHOLD for tip in all_fingertips
        # ) and max(
        #     hypot(first.x - second.x, first.y - second.y)
        #     for first, second in combinations(all_fingertips, 2)
        # ) / hand_size <= FINGERTIP_CLUSTER_THRESHOLD
        #
        # if fingertips_together:
        #     gesture = GestureState.FIST_CLOSE_WINDOW
        #     distance_ratio = None
        # else:
        if open_hand:
            gesture = GestureState.IDLE
            distance_ratio = None
        else:
            active_pinches = [
                item for item in pinch_ratios
                if item[1] <= self.pinch_threshold
            ]
            if len(active_pinches) == 2:
                active_pinches.sort(key=lambda item: item[1])
                if active_pinches[1][1] - active_pinches[0][1] < (
                    PINCH_DOMINANCE_MARGIN
                ):
                    gesture = GestureState.IDLE
                    distance_ratio = active_pinches[0][1]
                else:
                    gesture, distance_ratio = active_pinches[0]
            elif active_pinches:
                gesture, distance_ratio = active_pinches[0]
            else:
                gesture = GestureState.IDLE
                distance_ratio = (
                    min(pinch_ratios, key=lambda item: item[1])[1]
                    if pinch_ratios
                    else None
                )

        diagnostics = GestureDiagnostics(
            raw_gesture=gesture,
            curled_fingers=curled_fingers,
            visible_fingers=visible_fingers,
            curl_ratios=tuple(curl_ratios),
            finger_flex=tuple(finger_flex),
            left_pinch_ratio=left_pinch_ratio,
            right_pinch_ratio=right_pinch_ratio,
            open_hand=open_hand,
            index_only_hand=index_only_hand,
            two_finger_hand=two_finger_hand,
            ring_pinky_curled=ring_pinky_curled,
            pointer_pose=pointer_pose,
            palm_x=palm_x,
            palm_y=palm_y,
            wrist_x=wrist.x,
            wrist_y=wrist.y,
            hand_size=hand_size,
        )
        return GestureResult(gesture, distance_ratio), diagnostics
