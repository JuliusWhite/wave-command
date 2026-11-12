"""Geometric gesture rules that operate on detected hand landmarks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import hypot

from ..config import (
    CONFIDENCE_THRESHOLD,
    PINCH_THRESHOLD,
)
from .hand_detector import HandPrediction

FIST_CONFIRM_FRAMES = 3
FIST_RELEASE_FRAMES = 6
FIST_REARM_FRAMES = 3


class GestureState(str, Enum):
    IDLE = "IDLE"
    PINCH_LEFT_CLICK = "PINCH_LEFT_CLICK"
    PINCH_RIGHT_CLICK = "PINCH_RIGHT_CLICK"
    FIST_CLOSE_WINDOW = "FIST_CLOSE_WINDOW"


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


@dataclass(frozen=True)
class GestureResult:
    gesture: GestureState
    pinch_distance_ratio: float | None


class GestureEngine:
    """Evaluate geometric rules independently from model inference."""

    def __init__(self, pinch_threshold: float = PINCH_THRESHOLD) -> None:
        self.pinch_threshold = pinch_threshold
        self._stable_gesture = GestureState.IDLE
        self._fist_active = False
        self._fist_candidate_frames = 0
        self._fist_release_frames = 0
        self._fist_rearm_pending = False
        self._fist_rearm_frames = 0
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
        """Classify a hand, requiring a stable fist before triggering."""
        result, diagnostics = self._classify(hand)
        self._diagnostics = diagnostics
        curled_fingers = diagnostics.curled_fingers
        visible_fingers = diagnostics.visible_fingers

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
            if (
                result.gesture == GestureState.FIST_CLOSE_WINDOW
                or curled_fingers >= 3
            ):
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
            if curled_fingers < visible_fingers:
                self._fist_candidate_frames = 0
                return GestureResult(
                    self._stable_gesture,
                    result.pinch_distance_ratio,
                )

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

    def _classify(
        self,
        hand: HandPrediction | None,
    ) -> tuple[GestureResult, GestureDiagnostics]:
        """Classify finger curls and pinches normalized by hand size."""
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

        if curled_fingers >= 3 and curled_fingers == visible_fingers:
            gesture = GestureState.FIST_CLOSE_WINDOW
            distance_ratio = None
        else:
            active_pinches = [
                item for item in pinch_ratios if item[1] <= self.pinch_threshold
            ]
            if active_pinches:
                gesture, distance_ratio = min(
                    active_pinches,
                    key=lambda item: item[1],
                )
            elif curled_fingers >= 3:
                gesture = GestureState.FIST_CLOSE_WINDOW
                distance_ratio = None
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
        )
        return GestureResult(gesture, distance_ratio), diagnostics
