"""Map a normalized pointing-finger position to a smoothed screen cursor."""

from __future__ import annotations

import time
from collections.abc import Callable
from math import hypot, isfinite

from ..config import (
    POINTER_ACTIVE_MARGIN,
    POINTER_DEADZONE_PX,
    POINTER_MAX_SPEED_PX_PER_SEC,
    POINTER_MIRROR_X,
    POINTER_SCREEN_MARGIN_PX,
    POINTER_SMOOTHING,
)

_MAX_FRAME_INTERVAL_SEC = 1 / 15


def _screen_size() -> tuple[int, int]:
    import pyautogui

    width, height = pyautogui.size()
    return int(width), int(height)


def _move_cursor(x: int, y: int) -> None:
    import pyautogui

    pyautogui.moveTo(x, y)


def _cursor_position() -> tuple[int, int]:
    import pyautogui

    position = pyautogui.position()
    return int(position.x), int(position.y)


class PointerController:
    """Map webcam coordinates to a rate-limited, smoothed cursor position."""

    def __init__(
        self,
        move_cursor: Callable[[int, int], None] = _move_cursor,
        screen_size: Callable[[], tuple[int, int]] = _screen_size,
        active_margin: float = POINTER_ACTIVE_MARGIN,
        smoothing: float = POINTER_SMOOTHING,
        deadzone_px: float = POINTER_DEADZONE_PX,
        screen_margin_px: int = POINTER_SCREEN_MARGIN_PX,
        mirror_x: bool = POINTER_MIRROR_X,
        max_speed_px_per_sec: float = POINTER_MAX_SPEED_PX_PER_SEC,
        cursor_position: Callable[[], tuple[int, int]] = _cursor_position,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0 <= active_margin < 0.5:
            raise ValueError("active_margin must be between 0 and 0.5")
        if not 0 < smoothing <= 1:
            raise ValueError("smoothing must be greater than 0 and at most 1")
        if deadzone_px < 0 or screen_margin_px < 0:
            raise ValueError("pointer margins must be non-negative")
        if (
            not isfinite(max_speed_px_per_sec)
            or max_speed_px_per_sec <= 0
        ):
            raise ValueError("max_speed_px_per_sec must be finite and positive")

        self._move_cursor = move_cursor
        self._screen_size = screen_size
        self._cursor_position = cursor_position
        self._clock = clock
        self.active_margin = active_margin
        self.smoothing = smoothing
        self.deadzone_px = deadzone_px
        self.screen_margin_px = screen_margin_px
        self.mirror_x = mirror_x
        self.max_speed_px_per_sec = max_speed_px_per_sec
        self._last_position: tuple[float, float] | None = None
        self._last_update_at: float | None = None

    def move_to(self, normalized_x: float, normalized_y: float) -> None:
        """Map one pointing pose to the primary display."""
        if not isfinite(normalized_x) or not isfinite(normalized_y):
            raise ValueError("pointer coordinates must be finite")

        width, height = self._screen_size()
        if width <= 0 or height <= 0:
            raise RuntimeError("The primary display has invalid dimensions")

        if self.mirror_x:
            normalized_x = 1.0 - normalized_x
        normalized_x = min(
            1.0 - self.active_margin,
            max(self.active_margin, normalized_x),
        )
        normalized_y = min(
            1.0 - self.active_margin,
            max(self.active_margin, normalized_y),
        )

        screen_margin_x = min(self.screen_margin_px, max(0, (width - 1) // 2))
        screen_margin_y = min(self.screen_margin_px, max(0, (height - 1) // 2))
        target_x = screen_margin_x + round(
            (normalized_x - self.active_margin)
            / (1.0 - 2.0 * self.active_margin)
            * (width - 1 - 2 * screen_margin_x)
        )
        target_y = screen_margin_y + round(
            (normalized_y - self.active_margin)
            / (1.0 - 2.0 * self.active_margin)
            * (height - 1 - 2 * screen_margin_y)
        )

        now = self._clock()
        if self._last_position is None:
            cursor_x, cursor_y = self._cursor_position()
            self._last_position = (
                float(min(width - 1, max(0, cursor_x))),
                float(min(height - 1, max(0, cursor_y))),
            )
            self._last_update_at = now
            return

        last_x, last_y = self._last_position
        last_update_at = self._last_update_at
        elapsed = (
            min(_MAX_FRAME_INTERVAL_SEC, max(0.0, now - last_update_at))
            if last_update_at is not None
            else 0.0
        )
        self._last_update_at = now

        next_x = last_x + self.smoothing * (target_x - last_x)
        next_y = last_y + self.smoothing * (target_y - last_y)
        distance = hypot(next_x - last_x, next_y - last_y)
        if distance < self.deadzone_px:
            return

        max_distance = self.max_speed_px_per_sec * elapsed
        if max_distance <= 0:
            return
        if distance > max_distance:
            scale = max_distance / distance
            next_x = last_x + (next_x - last_x) * scale
            next_y = last_y + (next_y - last_y) * scale

        x, y = round(next_x), round(next_y)
        if (x, y) == (round(last_x), round(last_y)):
            return
        self._move_cursor(x, y)
        self._last_position = (next_x, next_y)

    def reset(self) -> None:
        """Reset smoothing after the pointing pose is released."""
        self._last_position = None
        self._last_update_at = None
