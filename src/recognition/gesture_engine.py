"""Geometric gesture rules that operate on detected hand landmarks."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
# from itertools import combinations
from math import acos, degrees, hypot

from ..config import (
    CONFIDENCE_THRESHOLD,
    PINCH_DOMINANCE_MARGIN,
    PINCH_THRESHOLD,
    SCROLL_DEAD_ZONE_RATIO_PER_SEC,
    SCROLL_MAX_HAND_SIZES_PER_SEC,
    SCROLL_SMOOTHING_ALPHA,
    SCROLL_START_DISTANCE_RATIO,
    SCROLL_UNITS_PER_HAND_SIZE_PER_SEC,
    HISTORY_SWIPE_DISTANCE_RATIO,
    TAB_SWIPE_DISTANCE_RATIO,
)
from .hand_detector import HandPrediction

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


@dataclass(frozen=True)
class GestureDiagnostics:
    raw_gesture: GestureState = GestureState.IDLE
    curled_fingers: int = 0
    visible_fingers: int = 0
    curl_ratios: tuple[float | None, ...] = (None, None, None, None)
    left_pinch_ratio: float | None = None
    right_pinch_ratio: float | None = None
    fist_confirmation_frames: int = 0
    fist_release_frames: int = 0
    fist_rearm_frames: int = 0
    fist_active: bool = False
    fist_rearm_pending: bool = False
    open_hand: bool = False
    pinch_enabled: bool = False
    index_only_hand: bool = False
    two_finger_hand: bool = False
    wrist_x: float | None = None
    wrist_y: float | None = None
    hand_size: float | None = None


@dataclass(frozen=True)
class GestureResult:
    gesture: GestureState
    pinch_distance_ratio: float | None
    scroll_amount: int = 0


class GestureEngine:
    """Evaluate geometric rules independently from model inference."""

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
        self._pinch_enabled = False
        self._pinch_waiting_for_open_hand_release = False
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
        self._history_swipe_origin_x: float | None = None
        self._history_swipe_origin_size: float | None = None
        self._tab_swipe_origin_x: float | None = None
        self._tab_swipe_origin_size: float | None = None
        self._tab_swipe_waiting_for_release = False
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
        """Classify pinches, open-palm scrolling, and two swipe gestures.

        Each pose owns one action: the open palm scrolls, an index-only hand
        navigates history, and an index-and-middle hand switches tabs.
        """
        result, diagnostics = self._classify(hand)
        scroll_amount = 0

        if (
            diagnostics.open_hand
            and diagnostics.wrist_x is not None
            and diagnostics.wrist_y is not None
            and diagnostics.hand_size is not None
        ):
            self._reset_history_swipe()
            self._reset_tab_swipe()
            scroll_amount = self._scroll_from_open_hand(
                diagnostics.wrist_x,
                diagnostics.wrist_y,
                diagnostics.hand_size,
            )
            if scroll_amount:
                # Scrolling consumes the pinch activation; the open palm must
                # be released and shown again before the next pinch.
                self._pinch_enabled = False
                self._pinch_waiting_for_open_hand_release = True
                scroll_gesture = (
                    GestureState.SCROLL_DOWN
                    if scroll_amount < 0
                    else GestureState.SCROLL_UP
                )
                result = GestureResult(scroll_gesture, None, scroll_amount)
                diagnostics = replace(
                    diagnostics,
                    raw_gesture=scroll_gesture,
                )
        elif (
            diagnostics.index_only_hand
            and diagnostics.wrist_x is not None
            and diagnostics.hand_size is not None
        ):
            result = GestureResult(GestureState.IDLE, None)
            diagnostics = replace(
                diagnostics,
                raw_gesture=GestureState.IDLE,
            )
            self._reset_scroll_tracking()
            self._reset_tab_swipe()
            history_gesture = self._history_from_index_only_hand(
                diagnostics.wrist_x,
                diagnostics.hand_size,
            )
            if history_gesture is not None:
                result = GestureResult(history_gesture, None)
                diagnostics = replace(
                    diagnostics,
                    raw_gesture=history_gesture,
                )
        elif (
            diagnostics.two_finger_hand
            and diagnostics.wrist_x is not None
            and diagnostics.hand_size is not None
        ):
            result = GestureResult(GestureState.IDLE, None)
            diagnostics = replace(
                diagnostics,
                raw_gesture=GestureState.IDLE,
            )
            self._reset_scroll_tracking()
            self._reset_history_swipe()
            tab_gesture = self._tab_from_two_finger_hand(
                diagnostics.wrist_x,
                diagnostics.hand_size,
            )
            if tab_gesture is not None:
                result = GestureResult(tab_gesture, None)
                diagnostics = replace(
                    diagnostics,
                    raw_gesture=tab_gesture,
                )
        else:
            self._reset_scroll_tracking()
            self._reset_history_swipe()
            self._reset_tab_swipe()

        if self._pinch_waiting_for_open_hand_release:
            if not diagnostics.open_hand:
                self._pinch_waiting_for_open_hand_release = False
        elif diagnostics.open_hand:
            self._pinch_enabled = True
        elif hand is None:
            self._pinch_enabled = False

        pinch_detected = result.gesture in {
            GestureState.PINCH_LEFT_CLICK,
            GestureState.PINCH_RIGHT_CLICK,
        }
        if not self._pinch_enabled and pinch_detected:
            result = GestureResult(
                GestureState.IDLE,
                result.pinch_distance_ratio,
            )
        elif pinch_detected:
            self._pinch_enabled = False
        diagnostics = replace(
            diagnostics,
            pinch_enabled=self._pinch_enabled,
        )
        self._diagnostics = diagnostics

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

    def _reset_history_swipe(self) -> None:
        self._history_swipe_origin_x = None
        self._history_swipe_origin_size = None

    def _reset_tab_swipe(self) -> None:
        self._tab_swipe_origin_x = None
        self._tab_swipe_origin_size = None
        self._tab_swipe_waiting_for_release = False

    def _tab_from_two_finger_hand(
        self,
        wrist_x: float,
        hand_size: float,
    ) -> GestureState | None:
        """Switch tabs once per two-finger pose.

        After a switch, the pose must be released before the next one, so
        bringing the hand back does not switch straight back again.
        """
        if self._tab_swipe_waiting_for_release:
            return None
        if (
            self._tab_swipe_origin_x is None
            or self._tab_swipe_origin_size is None
        ):
            self._tab_swipe_origin_x = wrist_x
            self._tab_swipe_origin_size = hand_size
            return None

        reference_size = (self._tab_swipe_origin_size + hand_size) / 2
        displacement_ratio = (self._tab_swipe_origin_x - wrist_x) / (
            reference_size
        )
        if abs(displacement_ratio) < TAB_SWIPE_DISTANCE_RATIO:
            return None

        self._tab_swipe_waiting_for_release = True
        return (
            GestureState.PREVIOUS_TAB
            if displacement_ratio > 0
            else GestureState.NEXT_TAB
        )

    def _history_from_index_only_hand(
        self,
        wrist_x: float,
        hand_size: float,
    ) -> GestureState | None:
        if (
            self._history_swipe_origin_x is None
            or self._history_swipe_origin_size is None
        ):
            self._history_swipe_origin_x = wrist_x
            self._history_swipe_origin_size = hand_size
            return None

        reference_size = (self._history_swipe_origin_size + hand_size) / 2
        displacement_ratio = (self._history_swipe_origin_x - wrist_x) / (
            reference_size
        )
        if abs(displacement_ratio) < HISTORY_SWIPE_DISTANCE_RATIO:
            return None

        self._history_swipe_origin_x = None
        self._history_swipe_origin_size = None
        return (
            GestureState.BACK
            if displacement_ratio > 0
            else GestureState.FORWARD
        )

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
        index_only_hand = (
            visible_fingers == len(finger_tips)
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
            left_pinch_ratio=left_pinch_ratio,
            right_pinch_ratio=right_pinch_ratio,
            open_hand=open_hand,
            index_only_hand=index_only_hand,
            two_finger_hand=two_finger_hand,
            wrist_x=wrist.x,
            wrist_y=wrist.y,
            hand_size=hand_size,
        )
        return GestureResult(gesture, distance_ratio), diagnostics
