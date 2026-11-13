"""Translate recognized gestures into mouse clicks and action messages."""

from __future__ import annotations

import ctypes
import sys
import time
from collections.abc import Callable
from typing import Literal

from ctypes import wintypes

from ..config import COMMAND_COOLDOWN_SEC, TAB_SWITCH_COOLDOWN_SEC
# from ..config import SCROLL_AMOUNT  # Scroll action disabled.
from ..recognition.gesture_engine import GestureState

MouseButton = Literal["left", "right"]
WM_SYSCOMMAND = 0x0112
SC_CLOSE = 0xF060


def _click_mouse(button: MouseButton) -> None:
    import pyautogui

    pyautogui.click(button=button)


def _scroll_mouse(amount: int) -> None:
    import pyautogui

    pyautogui.scroll(amount)


def _send_hotkey(*keys: str) -> None:
    import pyautogui

    pyautogui.hotkey(*keys)


def _close_current_window() -> None:
    if sys.platform == "win32":
        _request_windows_foreground_close()
        return

    import pyautogui

    pyautogui.hotkey("alt", "f4")


def _request_windows_foreground_close() -> None:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetForegroundWindow.argtypes = ()
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.PostMessageW.argtypes = (
        wintypes.HWND,
        wintypes.UINT,
        ctypes.c_size_t,
        ctypes.c_ssize_t,
    )
    user32.PostMessageW.restype = wintypes.BOOL

    window = user32.GetForegroundWindow()
    if not window:
        raise OSError("Windows has no foreground window to close")

    if not user32.PostMessageW(window, WM_SYSCOMMAND, SC_CLOSE, 0):
        raise OSError(
            ctypes.get_last_error(),
            "Windows rejected the close request for the foreground window",
        )


class CommandTranslator:
    """Translate enabled gestures into mouse actions and action messages."""

    ACTIONS = {
        GestureState.PINCH_LEFT_CLICK: "LEFT_CLICK",
        GestureState.PINCH_RIGHT_CLICK: "RIGHT_CLICK",
        GestureState.NEXT_TAB: "NEXT_TAB",
        GestureState.PREVIOUS_TAB: "PREVIOUS_TAB",
        GestureState.BACK: "BACK",
        GestureState.FORWARD: "FORWARD",
        # Scroll gestures disabled; retain mappings for possible reactivation.
        # GestureState.SCROLL_UP: "SCROLL_UP",
        # GestureState.SCROLL_DOWN: "SCROLL_DOWN",
        # Close-window action disabled; uncomment to restore.
        # GestureState.FIST_CLOSE_WINDOW: "CLOSE_WINDOW",
    }
    BUTTONS: dict[GestureState, MouseButton] = {
        GestureState.PINCH_LEFT_CLICK: "left",
        GestureState.PINCH_RIGHT_CLICK: "right",
    }
    # SCROLLS: dict[GestureState, int] = {
    #     GestureState.SCROLL_UP: 1,
    #     GestureState.SCROLL_DOWN: -1,
    # }

    def __init__(
        self,
        cooldown_sec: float = COMMAND_COOLDOWN_SEC,
        clock: Callable[[], float] = time.monotonic,
        emit: Callable[[str], None] = print,
        click: Callable[[MouseButton], None] = _click_mouse,
        close_window: Callable[[], None] = _close_current_window,
        scroll: Callable[[int], None] = _scroll_mouse,
        hotkey: Callable[..., None] = _send_hotkey,
    ) -> None:
        if cooldown_sec < 0:
            raise ValueError("cooldown_sec must be non-negative")
        self.cooldown_sec = cooldown_sec
        self._clock = clock
        self._emit = emit
        self._click = click
        self._scroll = scroll
        self._close_window = close_window
        self._hotkey = hotkey
        self._previous_gesture = GestureState.IDLE
        self._last_trigger_at = float("-inf")
        self._last_navigation_at = float("-inf")
        self._last_tab_switch_at = float("-inf")

    def process(self, gesture: GestureState) -> str | None:
        """Dispatch new actions with click and navigation cooldowns."""
        is_new_action = (
            gesture in self.ACTIONS and gesture != self._previous_gesture
        )
        self._previous_gesture = gesture

        if not is_new_action:
            return None

        now = self._clock()
        if now - self._last_tab_switch_at < TAB_SWITCH_COOLDOWN_SEC:
            return None
        if (
            gesture in self.BUTTONS
            and now - self._last_trigger_at < self.cooldown_sec
        ):
            return None
        if (
            gesture in {
                GestureState.NEXT_TAB,
                GestureState.PREVIOUS_TAB,
                GestureState.BACK,
                GestureState.FORWARD,
            }
            and now - self._last_navigation_at < TAB_SWITCH_COOLDOWN_SEC
        ):
            return None

        action = self.ACTIONS[gesture]
        # Retained for reactivation with the mapping above.
        # if gesture == GestureState.FIST_CLOSE_WINDOW:
        #     self._close_window()
        # else:
        # Scroll action disabled; retain for possible reactivation.
        # if gesture in self.SCROLLS:
        #     self._scroll(self.SCROLLS[gesture] * SCROLL_AMOUNT)
        if gesture == GestureState.NEXT_TAB:
            self._hotkey("ctrl", "tab")
            self._last_navigation_at = now
            self._last_tab_switch_at = now
        elif gesture == GestureState.PREVIOUS_TAB:
            self._hotkey("ctrl", "shift", "tab")
            self._last_navigation_at = now
            self._last_tab_switch_at = now
        elif gesture == GestureState.BACK:
            self._hotkey("alt", "left")
            self._last_navigation_at = now
        elif gesture == GestureState.FORWARD:
            self._hotkey("alt", "right")
            self._last_navigation_at = now
        else:
            self._click(self.BUTTONS[gesture])
            self._last_trigger_at = now
        # status = (
        #     "Requested"
        #     if gesture == GestureState.FIST_CLOSE_WINDOW
        #     else "Executed"
        # )
        status = "Executed"
        message = (
            "[WAVE COMMAND - TRANSLATOR] Gesture detected! -> "
            f"Action: {action} ({status})"
        )
        self._emit(message)
        return message
