"""Map a normalized pointing-finger position to a smoothed screen cursor."""

from __future__ import annotations

from collections.abc import Callable
from math import hypot, isfinite

from ..config import (
    POINTER_ACTIVE_MARGIN,
    POINTER_DEADZONE_PX,
    POINTER_MIRROR_X,
    POINTER_SCREEN_MARGIN_PX,
    POINTER_SMOOTHING,
)


def _screen_size() -> tuple[int, int]:
    import pyautogui

    width, height = pyautogui.size()
    return int(width), int(height)


def _move_cursor(x: int, y: int) -> None:
    import pyautogui

    pyautogui.moveTo(x, y)


class PointerController:
    """Convert webcam coordinates to cursor positions with smoothing."""

    def __init__(
        self,
        move_cursor: Callable[[int, int], None] = _move_cursor,
        screen_size: Callable[[], tuple[int, int]] = _screen_size,
        active_margin: float = POINTER_ACTIVE_MARGIN,
        smoothing: float = POINTER_SMOOTHING,
        deadzone_px: float = POINTER_DEADZONE_PX,
        screen_margin_px: int = POINTER_SCREEN_MARGIN_PX,
        mirror_x: bool = POINTER_MIRROR_X,
    ) -> None:
        if not 0 <= active_margin < 0.5:
            raise ValueError("active_margin must be between 0 and 0.5")
        if not 0 < smoothing <= 1:
            raise ValueError("smoothing must be greater than 0 and at most 1")
        if deadzone_px < 0 or screen_margin_px < 0:
            raise ValueError("pointer margins must be non-negative")

        self._move_cursor = move_cursor
        self._screen_size = screen_size
        self.active_margin = active_margin
        self.smoothing = smoothing
        self.deadzone_px = deadzone_px
        self.screen_margin_px = screen_margin_px
        self.mirror_x = mirror_x
        self._last_position: tuple[float, float] | None = None

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

        if self._last_position is None:
            next_x = float(target_x)
            next_y = float(target_y)
        else:
            last_x, last_y = self._last_position
            next_x = last_x + self.smoothing * (target_x - last_x)
            next_y = last_y + self.smoothing * (target_y - last_y)
            if hypot(next_x - last_x, next_y - last_y) < self.deadzone_px:
                return

        x, y = round(next_x), round(next_y)
        self._move_cursor(x, y)
        self._last_position = (next_x, next_y)

    def reset(self) -> None:
        """Reset smoothing after the pointing pose is released."""
        self._last_position = None
