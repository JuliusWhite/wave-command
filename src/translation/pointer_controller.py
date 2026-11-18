"""Move the OS cursor from palm positions reported by the gesture engine."""

from __future__ import annotations

import time
from collections.abc import Callable
from math import hypot, pi

from ..config import (
    POINTER_ACCEL_FULL_SPEED,
    POINTER_ACCEL_START_SPEED,
    POINTER_FAST_PX_PER_HAND_SIZE,
    POINTER_FILTER_BETA,
    POINTER_FILTER_DERIVATIVE_CUTOFF_HZ,
    POINTER_FILTER_MIN_CUTOFF_HZ,
    POINTER_INVERT_X,
    POINTER_SLOW_PX_PER_HAND_SIZE,
)
from ..recognition.gesture_engine import PointerSample
from .command_translator import _pyautogui


def _move_cursor_by(dx: int, dy: int) -> None:
    """Move the cursor relatively, staying one pixel inside the screen.

    PyAutoGUI's fail-safe aborts when the cursor reaches a screen corner.
    Keeping hand-driven movement off the edges leaves that fail-safe to the
    physical mouse: push it into a corner to stop Wave Command.
    """
    pyautogui = _pyautogui()
    width, height = pyautogui.size()
    x, y = pyautogui.position()
    target_x = min(max(x + dx, 1), width - 2)
    target_y = min(max(y + dy, 1), height - 2)
    if (target_x, target_y) != (x, y):
        pyautogui.moveTo(target_x, target_y)


class OneEuroFilter:
    """Adaptive low-pass filter: steady at rest, responsive when moving.

    See Casiez et al., "1 Euro Filter" (CHI 2012). The cutoff frequency rises
    with speed, trading jitter reduction for low lag during fast motion.
    """

    def __init__(
        self,
        min_cutoff: float = POINTER_FILTER_MIN_CUTOFF_HZ,
        beta: float = POINTER_FILTER_BETA,
        derivative_cutoff: float = POINTER_FILTER_DERIVATIVE_CUTOFF_HZ,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self._value: float | None = None
        self._derivative = 0.0

    @staticmethod
    def _alpha(cutoff: float, elapsed: float) -> float:
        tau = 1.0 / (2 * pi * cutoff)
        return 1.0 / (1.0 + tau / elapsed)

    def reset(self) -> None:
        self._value = None
        self._derivative = 0.0

    def __call__(self, value: float, elapsed: float) -> float:
        if self._value is None or elapsed <= 0:
            self._value = value if self._value is None else self._value
            return self._value
        derivative = (value - self._value) / elapsed
        derivative_alpha = self._alpha(self.derivative_cutoff, elapsed)
        self._derivative += derivative_alpha * (derivative - self._derivative)
        cutoff = self.min_cutoff + self.beta * abs(self._derivative)
        self._value += self._alpha(cutoff, elapsed) * (value - self._value)
        return self._value


class PointerController:
    """Turn filtered palm motion into accelerated, relative cursor motion.

    Movement is relative, like a mouse: when tracking stops (``None``), the
    cursor stays put and the next sample re-anchors without a jump. Frozen
    samples are treated the same way, so a click lands where the user was
    pointing and releasing the pinch does not drag the cursor.
    """

    def __init__(
        self,
        move_by: Callable[[int, int], None] = _move_cursor_by,
        clock: Callable[[], float] = time.monotonic,
        invert_x: bool = POINTER_INVERT_X,
        slow_px_per_hand_size: float = POINTER_SLOW_PX_PER_HAND_SIZE,
        fast_px_per_hand_size: float = POINTER_FAST_PX_PER_HAND_SIZE,
        accel_start_speed: float = POINTER_ACCEL_START_SPEED,
        accel_full_speed: float = POINTER_ACCEL_FULL_SPEED,
    ) -> None:
        if accel_full_speed <= accel_start_speed:
            raise ValueError("accel_full_speed must exceed accel_start_speed")
        self._move_by = move_by
        self._clock = clock
        self.invert_x = invert_x
        self.slow_px_per_hand_size = slow_px_per_hand_size
        self.fast_px_per_hand_size = fast_px_per_hand_size
        self.accel_start_speed = accel_start_speed
        self.accel_full_speed = accel_full_speed
        self._filter_x = OneEuroFilter()
        self._filter_y = OneEuroFilter()
        self._last_position: tuple[float, float] | None = None
        self._last_time: float | None = None
        self._remainder_x = 0.0
        self._remainder_y = 0.0

    def reset(self) -> None:
        """Stop tracking; the next sample re-anchors without moving."""
        self._filter_x.reset()
        self._filter_y.reset()
        self._last_position = None
        self._last_time = None
        self._remainder_x = 0.0
        self._remainder_y = 0.0

    def gain(self, speed: float) -> float:
        """Cursor pixels per hand size of palm travel at ``speed``."""
        progress = (speed - self.accel_start_speed) / (
            self.accel_full_speed - self.accel_start_speed
        )
        progress = min(1.0, max(0.0, progress))
        return self.slow_px_per_hand_size + progress * (
            self.fast_px_per_hand_size - self.slow_px_per_hand_size
        )

    def update(self, sample: PointerSample | None) -> None:
        # A freeze is handled like lifting the hand: the filter would
        # otherwise keep converging toward where the hand moved during the
        # pinch and drag the cursor off the target after the click.
        if sample is None or sample.frozen or sample.hand_size <= 1e-6:
            self.reset()
            return

        now = self._clock()
        if self._last_position is None or self._last_time is None:
            self._last_position = (
                self._filter_x(sample.x, 0.0),
                self._filter_y(sample.y, 0.0),
            )
            self._last_time = now
            return

        elapsed = now - self._last_time
        if elapsed <= 0:
            return
        x = self._filter_x(sample.x, elapsed)
        y = self._filter_y(sample.y, elapsed)
        last_x, last_y = self._last_position
        self._last_position = (x, y)
        self._last_time = now

        dx_hands = (x - last_x) / sample.hand_size
        dy_hands = (y - last_y) / sample.hand_size
        gain = self.gain(hypot(dx_hands, dy_hands) / elapsed)
        if self.invert_x:
            dx_hands = -dx_hands

        self._remainder_x += dx_hands * gain
        self._remainder_y += dy_hands * gain
        step_x = int(self._remainder_x)
        step_y = int(self._remainder_y)
        self._remainder_x -= step_x
        self._remainder_y -= step_y
        if step_x or step_y:
            self._move_by(step_x, step_y)
