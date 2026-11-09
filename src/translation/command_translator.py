"""Translate recognized gestures into safe, simulated command events."""

from __future__ import annotations

import time
from collections.abc import Callable

from ..config import COMMAND_COOLDOWN_SEC
from ..recognition.gesture_engine import GestureState


class CommandTranslator:
    """Map gestures to action labels without issuing operating-system input."""

    ACTIONS = {
        GestureState.PINCH_CLOSE_WINDOW: "CLOSE_WINDOW",
    }

    def __init__(
        self,
        cooldown_sec: float = COMMAND_COOLDOWN_SEC,
        clock: Callable[[], float] = time.monotonic,
        emit: Callable[[str], None] = print,
    ) -> None:
        if cooldown_sec < 0:
            raise ValueError("cooldown_sec must be non-negative")
        self.cooldown_sec = cooldown_sec
        self._clock = clock
        self._emit = emit
        self._previous_gesture = GestureState.IDLE
        self._last_trigger_at = float("-inf")

    def process(self, gesture: GestureState) -> str | None:
        """Emit an action once per pinch transition, subject to cooldown."""
        is_new_pinch = (
            gesture == GestureState.PINCH_CLOSE_WINDOW
            and self._previous_gesture != GestureState.PINCH_CLOSE_WINDOW
        )
        self._previous_gesture = gesture

        if not is_new_pinch:
            return None

        now = self._clock()
        if now - self._last_trigger_at < self.cooldown_sec:
            return None

        action = self.ACTIONS[gesture]
        message = (
            "[WAVE COMMAND - TRANSLATOR] Pinch Detected! -> "
            f"Action: {action} (Simulated)"
        )
        self._emit(message)
        self._last_trigger_at = now
        return message

