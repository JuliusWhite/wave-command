import sys
import unittest
from unittest.mock import Mock, patch

from src.recognition.gesture_engine import GestureState
from src.translation.command_translator import (
    CommandTranslator,
    _close_current_window,
    _request_windows_foreground_close,
    _scroll_mouse,
)


class CommandTranslatorTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "win32", "Windows API only")
    def test_close_window_targets_the_foreground_window(self) -> None:
        post_message = Mock(return_value=1)
        user32 = Mock(
            GetForegroundWindow=Mock(return_value=12345),
            PostMessageW=post_message,
        )
        with patch("ctypes.WinDLL", return_value=user32):
            _request_windows_foreground_close()

        self.assertEqual(
            post_message.call_args.args,
            (12345, 0x0112, 0xF060, 0),
        )

    @unittest.skipUnless(sys.platform == "win32", "Windows API only")
    def test_close_window_reports_missing_foreground_window(self) -> None:
        post_message = Mock()
        user32 = Mock(
            GetForegroundWindow=Mock(return_value=0),
            PostMessageW=post_message,
        )
        with patch("ctypes.WinDLL", return_value=user32):
            with self.assertRaisesRegex(OSError, "no foreground window"):
                _request_windows_foreground_close()

        post_message.assert_not_called()

    @unittest.skipUnless(sys.platform == "win32", "Windows API only")
    def test_close_window_reports_rejected_request(self) -> None:
        user32 = Mock(
            GetForegroundWindow=Mock(return_value=12345),
            PostMessageW=Mock(return_value=0),
        )
        with patch("ctypes.WinDLL", return_value=user32):
            with self.assertRaisesRegex(OSError, "rejected the close request"):
                _request_windows_foreground_close()

    @unittest.skipUnless(sys.platform != "win32", "Non-Windows fallback only")
    def test_close_window_sends_alt_f4_on_non_windows(self) -> None:
        hotkey = Mock()
        with patch.dict("sys.modules", {"pyautogui": Mock(hotkey=hotkey)}):
            _close_current_window()

        hotkey.assert_called_once_with("alt", "f4")

    def test_left_click_triggers_once_and_rearms_after_idle(self) -> None:
        messages: list[str] = []
        buttons: list[str] = []
        current_time = [10.0]
        translator = CommandTranslator(
            cooldown_sec=0,
            clock=lambda: current_time[0],
            emit=messages.append,
            click=buttons.append,
        )

        first = translator.process(GestureState.PINCH_LEFT_CLICK)
        held = translator.process(GestureState.PINCH_LEFT_CLICK)
        translator.process(GestureState.IDLE)
        current_time[0] = 12.0  # past the double-click window
        second = translator.process(GestureState.PINCH_LEFT_CLICK)

        self.assertIsNotNone(first)
        self.assertIsNone(held)
        self.assertIsNotNone(second)
        self.assertEqual(len(messages), 2)
        self.assertIn("LEFT_CLICK (Executed)", messages[0])
        self.assertEqual(buttons, ["left", "left"])

    def test_right_click_uses_right_mouse_button(self) -> None:
        buttons: list[str] = []
        messages: list[str] = []
        translator = CommandTranslator(
            cooldown_sec=0,
            emit=messages.append,
            click=buttons.append,
        )

        translator.process(GestureState.PINCH_RIGHT_CLICK)

        self.assertEqual(buttons, ["right"])
        self.assertIn("RIGHT_CLICK (Executed)", messages[0])

    def test_scroll_direction_uses_configured_amount_without_click_cooldown(
        self,
    ) -> None:
        scrolls: list[int] = []
        messages: list[str] = []
        translator = CommandTranslator(
            cooldown_sec=10,
            emit=messages.append,
            scroll=scrolls.append,
        )

        first = translator.process(
            GestureState.SCROLL_DOWN,
            scroll_amount=-2,
        )
        repeated = translator.process(
            GestureState.SCROLL_DOWN,
            scroll_amount=-1,
        )
        translator.process(GestureState.IDLE)
        reverse = translator.process(
            GestureState.SCROLL_UP,
            scroll_amount=1,
        )

        self.assertEqual(scrolls, [-2, -1, 1])
        self.assertIsNotNone(first)
        self.assertIsNone(repeated)
        self.assertIsNotNone(reverse)
        self.assertEqual(len(messages), 2)

    def test_tab_swipes_send_browser_navigation_shortcuts(self) -> None:
        hotkeys: list[tuple[str, ...]] = []
        current_time = [10.0]
        translator = CommandTranslator(
            cooldown_sec=10,
            clock=lambda: current_time[0],
            hotkey=lambda *keys: hotkeys.append(keys),
        )

        translator.process(GestureState.NEXT_TAB)
        translator.process(GestureState.IDLE)
        current_time[0] += 1.0
        translator.process(GestureState.PREVIOUS_TAB)

        self.assertEqual(
            hotkeys,
            [("ctrl", "tab"), ("ctrl", "shift", "tab")],
        )

    def test_window_swipes_send_window_switch_shortcuts(self) -> None:
        hotkeys: list[tuple[str, ...]] = []
        current_time = [10.0]
        translator = CommandTranslator(
            cooldown_sec=10,
            clock=lambda: current_time[0],
            switch_window=lambda *keys: hotkeys.append(keys),
        )

        translator.process(GestureState.NEXT_WINDOW)
        translator.process(GestureState.IDLE)
        current_time[0] += 0.5
        translator.process(GestureState.PREVIOUS_WINDOW)  # cooling down
        translator.process(GestureState.IDLE)
        current_time[0] += 0.5
        translator.process(GestureState.PREVIOUS_WINDOW)

        self.assertEqual(
            hotkeys,
            [("alt", "tab"), ("alt", "shift", "tab")],
        )

    def test_tab_cooldown_blocks_repeat_and_opposite_swipes(self) -> None:
        hotkeys: list[tuple[str, ...]] = []
        current_time = [10.0]
        translator = CommandTranslator(
            cooldown_sec=0,
            clock=lambda: current_time[0],
            hotkey=lambda *keys: hotkeys.append(keys),
        )

        translator.process(GestureState.NEXT_TAB)
        translator.process(GestureState.IDLE)
        current_time[0] = 10.5
        translator.process(GestureState.PREVIOUS_TAB)
        translator.process(GestureState.IDLE)
        current_time[0] = 10.75
        translator.process(GestureState.NEXT_TAB)
        translator.process(GestureState.IDLE)
        current_time[0] = 11.0
        translator.process(GestureState.PREVIOUS_TAB)

        self.assertEqual(
            hotkeys,
            [("ctrl", "tab"), ("ctrl", "shift", "tab")],
        )

    def test_history_swipes_send_alt_arrows_and_share_navigation_cooldown(
        self,
    ) -> None:
        hotkeys: list[tuple[str, ...]] = []
        current_time = [10.0]
        translator = CommandTranslator(
            cooldown_sec=0,
            clock=lambda: current_time[0],
            hotkey=lambda *keys: hotkeys.append(keys),
        )

        translator.process(GestureState.NEXT_TAB)
        translator.process(GestureState.IDLE)
        current_time[0] = 10.5
        translator.process(GestureState.BACK)
        translator.process(GestureState.IDLE)
        current_time[0] = 11.0
        translator.process(GestureState.BACK)
        translator.process(GestureState.IDLE)
        current_time[0] = 12.0
        translator.process(GestureState.FORWARD)

        self.assertEqual(
            hotkeys,
            [("ctrl", "tab"), ("alt", "left"), ("alt", "right")],
        )

    def test_tab_switch_cooldown_suppresses_a_following_pinch(self) -> None:
        hotkeys: list[tuple[str, ...]] = []
        buttons: list[str] = []
        current_time = [10.0]
        translator = CommandTranslator(
            cooldown_sec=0,
            clock=lambda: current_time[0],
            click=buttons.append,
            hotkey=lambda *keys: hotkeys.append(keys),
        )

        translator.process(GestureState.NEXT_TAB)
        translator.process(GestureState.IDLE)
        current_time[0] = 10.5
        self.assertIsNone(
            translator.process(GestureState.PINCH_LEFT_CLICK)
        )
        translator.process(GestureState.IDLE)
        current_time[0] = 11.0
        self.assertIsNotNone(
            translator.process(GestureState.PINCH_LEFT_CLICK)
        )

        self.assertEqual(hotkeys, [("ctrl", "tab")])
        self.assertEqual(buttons, ["left"])

    def test_scroll_action_dispatches_mouse_wheel(self) -> None:
        scrolls: list[int] = []
        translator = CommandTranslator(scroll=scrolls.append)

        result = translator.process(
            GestureState.SCROLL_DOWN,
            scroll_amount=-1,
        )

        self.assertIsNotNone(result)
        self.assertEqual(scrolls, [-1])

    @unittest.skip("Close-window action temporarily disabled")
    def test_fist_closes_window_without_clicking(self) -> None:
        messages: list[str] = []
        buttons: list[str] = []
        close_calls: list[bool] = []
        translator = CommandTranslator(
            cooldown_sec=0,
            emit=messages.append,
            click=buttons.append,
            close_window=lambda: close_calls.append(True),
        )

        action = translator.process(GestureState.FIST_CLOSE_WINDOW)
        held = translator.process(GestureState.FIST_CLOSE_WINDOW)

        self.assertIsNotNone(action)
        self.assertIsNone(held)
        self.assertEqual(close_calls, [True])
        self.assertEqual(buttons, [])
        self.assertIn("CLOSE_WINDOW (Requested)", messages[0])

    @unittest.skip("Close-window action temporarily disabled")
    def test_fist_close_is_not_blocked_by_click_cooldown(self) -> None:
        close_calls: list[bool] = []
        current_time = [10.0]
        translator = CommandTranslator(
            cooldown_sec=1.5,
            clock=lambda: current_time[0],
            click=lambda button: None,
            close_window=lambda: close_calls.append(True),
        )

        translator.process(GestureState.PINCH_LEFT_CLICK)
        translator.process(GestureState.IDLE)
        self.assertIsNotNone(
            translator.process(GestureState.FIST_CLOSE_WINDOW)
        )

        self.assertEqual(close_calls, [True])

    def test_middle_finger_closes_window_once_and_survives_errors(self) -> None:
        messages: list[str] = []
        close = Mock(side_effect=[None, OSError("nope")])
        translator = CommandTranslator(
            emit=messages.append, close_window=close
        )
        gesture = GestureState.MIDDLE_FINGER_CLOSE_WINDOW

        self.assertIn("CLOSE_WINDOW (Requested)", translator.process(gesture))
        self.assertIsNone(translator.process(gesture))
        close.assert_called_once_with()
        translator.process(GestureState.IDLE)
        self.assertIsNone(translator.process(gesture))
        self.assertIn("nope", messages[-1])

    @unittest.skipUnless(sys.platform == "win32", "Windows API only")
    def test_close_window_never_targets_the_desktop(self) -> None:
        def class_name(_window, buffer, _size):
            buffer.value = "Progman"
            return 7

        user32 = Mock(
            GetForegroundWindow=Mock(return_value=1),
            GetClassNameW=Mock(side_effect=class_name),
        )
        with patch("ctypes.WinDLL", return_value=user32):
            with self.assertRaisesRegex(OSError, "desktop"):
                _request_windows_foreground_close()
        user32.PostMessageW.assert_not_called()

    def test_fist_close_window_action_is_disabled(self) -> None:
        close_calls: list[bool] = []
        translator = CommandTranslator(
            cooldown_sec=0,
            close_window=lambda: close_calls.append(True),
        )

        result = translator.process(GestureState.FIST_CLOSE_WINDOW)

        self.assertIsNone(result)
        self.assertEqual(close_calls, [])

    def test_cooldown_suppresses_new_pinch_transition(self) -> None:
        messages: list[str] = []
        buttons: list[str] = []
        current_time = [10.0]
        translator = CommandTranslator(
            cooldown_sec=1.5,
            clock=lambda: current_time[0],
            emit=messages.append,
            click=buttons.append,
        )

        translator.process(GestureState.PINCH_LEFT_CLICK)
        translator.process(GestureState.IDLE)
        # Past the double-click window, but still within the click cooldown.
        current_time[0] = 11.25
        self.assertIsNone(translator.process(GestureState.PINCH_LEFT_CLICK))
        self.assertEqual(len(messages), 1)
        self.assertEqual(buttons, ["left"])

    def test_second_left_pinch_within_window_double_clicks(self) -> None:
        messages: list[str] = []
        clicks: list[str] = []
        current_time = [10.0]
        translator = CommandTranslator(
            cooldown_sec=1.5,
            clock=lambda: current_time[0],
            emit=messages.append,
            click=lambda button: clicks.append(button),
            double_click=lambda button: clicks.append(f"double-{button}"),
        )

        translator.process(GestureState.PINCH_LEFT_CLICK)
        translator.process(GestureState.IDLE)
        current_time[0] = 10.8
        double = translator.process(GestureState.PINCH_LEFT_CLICK)
        translator.process(GestureState.IDLE)
        # A third quick pinch does not double-click again; the click
        # cooldown applies from the double-click.
        current_time[0] = 11.2
        third = translator.process(GestureState.PINCH_LEFT_CLICK)

        self.assertEqual(clicks, ["left", "double-left"])
        self.assertIsNotNone(double)
        self.assertIn("DOUBLE_CLICK (Executed)", messages[1])
        self.assertIsNone(third)

    def test_right_click_never_starts_a_double_click(self) -> None:
        clicks: list[str] = []
        current_time = [10.0]
        translator = CommandTranslator(
            cooldown_sec=0,
            clock=lambda: current_time[0],
            emit=lambda _message: None,
            click=lambda button: clicks.append(button),
            double_click=lambda button: clicks.append(f"double-{button}"),
        )

        translator.process(GestureState.PINCH_RIGHT_CLICK)
        translator.process(GestureState.IDLE)
        current_time[0] = 10.5
        translator.process(GestureState.PINCH_RIGHT_CLICK)
        translator.process(GestureState.IDLE)
        current_time[0] = 11.0
        translator.process(GestureState.PINCH_LEFT_CLICK)

        self.assertEqual(clicks, ["right", "right", "left"])

    def test_gesture_held_through_bypass_does_not_fire_on_resume(
        self,
    ) -> None:
        buttons: list[str] = []
        translator = CommandTranslator(
            cooldown_sec=0, emit=lambda _message: None, click=buttons.append
        )

        translator.observe(GestureState.PINCH_LEFT_CLICK)
        held = translator.process(GestureState.PINCH_LEFT_CLICK)
        translator.process(GestureState.IDLE)
        fresh = translator.process(GestureState.PINCH_LEFT_CLICK)

        self.assertIsNone(held)
        self.assertIsNotNone(fresh)
        self.assertEqual(buttons, ["left"])



class PyAutoGuiDispatchTests(unittest.TestCase):
    def test_dispatch_disables_pyautogui_pause(self) -> None:
        fake_pyautogui = Mock()
        fake_pyautogui.PAUSE = 0.1

        with patch.dict(sys.modules, {"pyautogui": fake_pyautogui}):
            _scroll_mouse(3)

        self.assertEqual(fake_pyautogui.PAUSE, 0.0)
        fake_pyautogui.scroll.assert_called_once_with(3)


if __name__ == "__main__":
    unittest.main()
