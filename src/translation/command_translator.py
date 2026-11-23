"""Translate recognized gestures into mouse clicks and action messages."""

from __future__ import annotations

import ctypes
import sys
import time
from collections.abc import Callable
from typing import Literal

from ctypes import wintypes

from ..config import (
    COMMAND_COOLDOWN_SEC,
    DOUBLE_CLICK_WINDOW_SEC,
    TAB_SWITCH_COOLDOWN_SEC,
)
from ..recognition.gesture_engine import GestureState

MouseButton = Literal["left", "right"]
WM_SYSCOMMAND = 0x0112
SC_CLOSE = 0xF060
# Desktop and taskbar windows; closing these would open the shutdown dialog.
SHELL_WINDOW_CLASSES = frozenset(
    {"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"}
)


def _pyautogui():
    """Import PyAutoGUI lazily and disable its per-call sleep.

    PyAutoGUI sleeps for ``PAUSE`` seconds (0.1 by default) after every call.
    Scrolling dispatches on consecutive frames, so that delay would block the
    camera loop and drop the frame rate while the user scrolls.
    """
    import pyautogui

    pyautogui.PAUSE = 0.0
    return pyautogui


def _click_mouse(button: MouseButton) -> None:
    _pyautogui().click(button=button)


def _double_click_mouse(button: MouseButton) -> None:
    _pyautogui().doubleClick(button=button)


def _scroll_mouse(amount: int) -> None:
    _pyautogui().scroll(amount)


def _send_hotkey(*keys: str) -> None:
    _pyautogui().hotkey(*keys)


WINDOW_SWITCH_KEY_HOLD_SEC = 0.05


def _send_window_switch(*keys: str) -> None:
    """Press a window-switch shortcut like a person would.

    Windows ignores an Alt+Tab whose keys go down and up in the same
    instant, so the modifiers are held briefly around the Tab press.
    """
    pyautogui = _pyautogui()
    *modifiers, key = keys
    for modifier in modifiers:
        pyautogui.keyDown(modifier)
    try:
        time.sleep(WINDOW_SWITCH_KEY_HOLD_SEC)
        pyautogui.press(key)
        time.sleep(WINDOW_SWITCH_KEY_HOLD_SEC)
    finally:
        for modifier in reversed(modifiers):
            pyautogui.keyUp(modifier)


def _close_current_window() -> None:
    if sys.platform == "win32":
        _request_windows_foreground_close()
        return

    _pyautogui().hotkey("alt", "f4")


def _request_windows_foreground_close() -> None:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetForegroundWindow.argtypes = ()
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetClassNameW.argtypes = (
        wintypes.HWND,
        wintypes.LPWSTR,
        ctypes.c_int,
    )
    user32.GetClassNameW.restype = ctypes.c_int
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

    class_name = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(window, class_name, len(class_name))
    if class_name.value in SHELL_WINDOW_CLASSES:
        raise OSError("The desktop or taskbar is focused; nothing to close")

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
        GestureState.NEXT_WINDOW: "NEXT_WINDOW",
        GestureState.PREVIOUS_WINDOW: "PREVIOUS_WINDOW",
        GestureState.BACK: "BACK",
        GestureState.FORWARD: "FORWARD",
        GestureState.SCROLL_UP: "SCROLL_UP",
        GestureState.SCROLL_DOWN: "SCROLL_DOWN",
        GestureState.MIDDLE_FINGER_CLOSE_WINDOW: "CLOSE_WINDOW",
        # Fist close-window action disabled; uncomment to restore.
        # GestureState.FIST_CLOSE_WINDOW: "CLOSE_WINDOW",
    }
    BUTTONS: dict[GestureState, MouseButton] = {
        GestureState.PINCH_LEFT_CLICK: "left",
        GestureState.PINCH_RIGHT_CLICK: "right",
    }
    SCROLLS: dict[GestureState, int] = {
        GestureState.SCROLL_UP: 1,
        GestureState.SCROLL_DOWN: -1,
    }

    def __init__(
        self,
        cooldown_sec: float = COMMAND_COOLDOWN_SEC,
        clock: Callable[[], float] = time.monotonic,
        emit: Callable[[str], None] = print,
        click: Callable[[MouseButton], None] = _click_mouse,
        close_window: Callable[[], None] = _close_current_window,
        scroll: Callable[[int], None] = _scroll_mouse,
        hotkey: Callable[..., None] = _send_hotkey,
        switch_window: Callable[..., None] = _send_window_switch,
        double_click: Callable[[MouseButton], None] = _double_click_mouse,
        double_click_window_sec: float = DOUBLE_CLICK_WINDOW_SEC,
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
        self._switch_window = switch_window
        self._double_click = double_click
        self.double_click_window_sec = double_click_window_sec
        # Time of the last single left click, which a second left pinch can
        # turn into a double-click.
        self._last_left_click_at = float("-inf")
        self._previous_gesture = GestureState.IDLE
        self._last_trigger_at = float("-inf")
        self._last_navigation_at = float("-inf")
        self._last_tab_switch_at = float("-inf")

    def observe(self, gesture: GestureState) -> None:
        """Track a gesture without acting on it (e.g. while bypassed).

        A gesture still held when actions resume is then not treated as new,
        so resuming never fires a stale click or swipe.
        """
        self._previous_gesture = gesture

    def process(
        self,
        gesture: GestureState,
        scroll_amount: int = 0,
    ) -> str | None:
        """Dispatch new actions with click and navigation cooldowns."""
        previous_gesture = self._previous_gesture
        is_new_action = (
            gesture in self.ACTIONS
            and (
                gesture != previous_gesture
                or (gesture in self.SCROLLS and scroll_amount != 0)
            )
        )
        self._previous_gesture = gesture

        if not is_new_action:
            return None

        now = self._clock()
        if (
            gesture not in self.SCROLLS
            and now - self._last_tab_switch_at < TAB_SWITCH_COOLDOWN_SEC
        ):
            return None
        is_double_click = (
            gesture == GestureState.PINCH_LEFT_CLICK
            and now - self._last_left_click_at <= self.double_click_window_sec
        )
        if (
            gesture in self.BUTTONS
            and not is_double_click
            and now - self._last_trigger_at < self.cooldown_sec
        ):
            return None
        if (
            gesture in {
                GestureState.NEXT_TAB,
                GestureState.PREVIOUS_TAB,
                GestureState.NEXT_WINDOW,
                GestureState.PREVIOUS_WINDOW,
                GestureState.BACK,
                GestureState.FORWARD,
            }
            and now - self._last_navigation_at < TAB_SWITCH_COOLDOWN_SEC
        ):
            return None

        action = self.ACTIONS[gesture]
        status = "Executed"
        # Retained for reactivation with the mapping above.
        # if gesture == GestureState.FIST_CLOSE_WINDOW:
        #     self._close_window()
        # else:
        if gesture in self.SCROLLS:
            if scroll_amount == 0:
                return None
            self._scroll(scroll_amount)
            if gesture == previous_gesture:
                return None
        elif gesture == GestureState.MIDDLE_FINGER_CLOSE_WINDOW:
            # The app may still ask to save, so this is only a request.
            status = "Requested"
            try:
                self._close_window()
            except OSError as error:
                self._emit(
                    "[WAVE COMMAND - TRANSLATOR] Could not close the window: "
                    f"{error}"
                )
                return None
        elif gesture == GestureState.NEXT_TAB:
            self._hotkey("ctrl", "tab")
            self._last_navigation_at = now
            self._last_tab_switch_at = now
        elif gesture == GestureState.PREVIOUS_TAB:
            self._hotkey("ctrl", "shift", "tab")
            self._last_navigation_at = now
            self._last_tab_switch_at = now
        elif gesture == GestureState.NEXT_WINDOW:
            self._switch_window("alt", "tab")
            self._last_navigation_at = now
            self._last_tab_switch_at = now
        elif gesture == GestureState.PREVIOUS_WINDOW:
            self._switch_window("alt", "shift", "tab")
            self._last_navigation_at = now
            self._last_tab_switch_at = now
        elif gesture == GestureState.BACK:
            self._hotkey("alt", "left")
            self._last_navigation_at = now
        elif gesture == GestureState.FORWARD:
            self._hotkey("alt", "right")
            self._last_navigation_at = now
        elif is_double_click:
            # Both clicks go out together at the same spot; the first pinch's
            # click before them only selects, as with a real mouse.
            self._double_click("left")
            action = "DOUBLE_CLICK"
            self._last_trigger_at = now
            # A third quick pinch is a new single click, after the cooldown.
            self._last_left_click_at = float("-inf")
        else:
            self._click(self.BUTTONS[gesture])
            self._last_trigger_at = now
            self._last_left_click_at = (
                now
                if gesture == GestureState.PINCH_LEFT_CLICK
                else float("-inf")
            )
        # status = (
        #     "Requested"
        #     if gesture == GestureState.FIST_CLOSE_WINDOW
        #     else "Executed"
        # )
        message = (
            "[WAVE COMMAND - TRANSLATOR] Gesture detected! -> "
            f"Action: {action} ({status})"
        )
        self._emit(message)
        return message
