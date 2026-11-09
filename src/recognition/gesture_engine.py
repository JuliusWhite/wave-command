"""Geometric gesture rules that operate on detected hand landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import hypot

from ..config import CONFIDENCE_THRESHOLD, PINCH_THRESHOLD
from .hand_detector import HandPrediction


class GestureState(str, Enum):
    IDLE = "IDLE"
    PINCH_CLOSE_WINDOW = "PINCH_CLOSE_WINDOW"


@dataclass(frozen=True)
class GestureResult:
    gesture: GestureState
    pinch_distance_ratio: float | None


class GestureEngine:
    """Evaluate geometric rules independently from model inference."""

    def __init__(self, pinch_threshold: float = PINCH_THRESHOLD) -> None:
        self.pinch_threshold = pinch_threshold

    def evaluate(self, hand: HandPrediction | None) -> GestureResult:
        """Classify a hand using tip distance normalized by hand size."""
        if hand is None or len(hand.keypoints) != 21:
            return GestureResult(GestureState.IDLE, None)

        thumb_tip = hand.keypoints[4]
        index_tip = hand.keypoints[8]
        wrist = hand.keypoints[0]
        middle_mcp = hand.keypoints[9]

        if min(
            thumb_tip.conf,
            index_tip.conf,
            wrist.conf,
            middle_mcp.conf,
        ) < CONFIDENCE_THRESHOLD:
            return GestureResult(GestureState.IDLE, None)

        hand_size = hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y)
        if hand_size <= 1e-6:
            return GestureResult(GestureState.IDLE, None)

        tip_distance = hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
        distance_ratio = tip_distance / hand_size
        gesture = (
            GestureState.PINCH_CLOSE_WINDOW
            if distance_ratio <= self.pinch_threshold
            else GestureState.IDLE
        )
        return GestureResult(gesture, distance_ratio)
