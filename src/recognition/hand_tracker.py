"""Keep control with one hand when several are in view.

The detector returns hands in no stable order, so with two hands on screen
"the first hand" can swap between frames and gestures jump between them.
``HandTracker`` locks onto the first hand that appears and follows it frame
to frame by palm position. Other hands are ignored until the locked hand
has been gone for a few frames; then the next hand in view takes over.
"""

from __future__ import annotations

import math

from ..config import HAND_TRACK_LOST_FRAMES, HAND_TRACK_MAX_JUMP
from .hand_detector import HandPrediction

# Wrist and the four finger bases: a palm center that ignores finger poses.
_PALM_LANDMARKS = (0, 5, 9, 13, 17)


def palm_center(hand: HandPrediction) -> tuple[float, float]:
    """Palm center in normalized image coordinates (0..1)."""
    points = [hand.keypoints[index] for index in _PALM_LANDMARKS]
    return (
        sum(point.normalized_x for point in points) / len(points),
        sum(point.normalized_y for point in points) / len(points),
    )


class HandTracker:
    """Follow the first hand that came into view until it leaves."""

    def __init__(
        self,
        max_jump: float = HAND_TRACK_MAX_JUMP,
        lost_frames: int = HAND_TRACK_LOST_FRAMES,
    ) -> None:
        self.max_jump = max_jump
        self.lost_frames = lost_frames
        self._position: tuple[float, float] | None = None
        self._missing_frames = 0

    @property
    def locked(self) -> bool:
        return self._position is not None

    def reset(self) -> None:
        self._position = None
        self._missing_frames = 0

    def update(
        self, hands: list[HandPrediction]
    ) -> tuple[HandPrediction | None, list[HandPrediction]]:
        """Return (tracked hand or None, the ignored hands) for one frame."""
        if self._position is None:
            if not hands:
                return None, []
            # Nothing locked yet: take the first hand the detector reports.
            tracked = hands[0]
        else:
            tracked = self._nearest(hands)
            if tracked is None:
                # The locked hand is not seen. Keep the lock for a few frames
                # so a detection dropout does not hand control to another hand.
                self._missing_frames += 1
                if self._missing_frames > self.lost_frames:
                    self.reset()
                return None, list(hands)

        self._position = palm_center(tracked)
        self._missing_frames = 0
        return tracked, [hand for hand in hands if hand is not tracked]

    def _nearest(self, hands: list[HandPrediction]) -> HandPrediction | None:
        """The hand closest to the locked position, if close enough."""
        assert self._position is not None
        best: HandPrediction | None = None
        best_distance = self.max_jump
        for hand in hands:
            x, y = palm_center(hand)
            distance = math.hypot(x - self._position[0], y - self._position[1])
            if distance <= best_distance:
                best, best_distance = hand, distance
        return best
