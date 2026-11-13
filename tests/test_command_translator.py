import sys
import unittest
from unittest.mock import Mock, patch

from src.recognition.gesture_engine import GestureState
from src.translation.command_translator import (
    CommandTranslator,
    _close_current_window,
    _request_windows_foreground_close,
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
        translator = CommandTranslator(
            cooldown_sec=0,
            emit=messages.append,
            click=buttons.append,
        )

        first = translator.process(GestureState.PINCH_LEFT_CLICK)
        held = translator.process(GestureState.PINCH_LEFT_CLICK)
        translator.process(GestureState.IDLE)
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

    @unittest.skip("Scroll actions temporarily disabled")
    def test_scroll_direction_uses_configured_amount_without_click_cooldown(
        self,
    ) -> None:
        scrolls: list[int] = []
        translator = CommandTranslator(
            cooldown_sec=10,
            scroll=scrolls.append,
        )

        translator.process(GestureState.SCROLL_DOWN)
        translator.process(GestureState.IDLE)
        translator.process(GestureState.SCROLL_UP)

        self.assertEqual(scrolls, [-10, 10])

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

    def test_scroll_action_is_disabled(self) -> None:
        scrolls: list[int] = []
        translator = CommandTranslator(scroll=scrolls.append)

        result = translator.process(GestureState.SCROLL_DOWN)

        self.assertIsNone(result)
        self.assertEqual(scrolls, [])

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

    def test_close_window_action_is_disabled(self) -> None:
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
        current_time[0] = 11.0
        self.assertIsNone(translator.process(GestureState.PINCH_LEFT_CLICK))
        self.assertEqual(len(messages), 1)
        self.assertEqual(buttons, ["left"])


if __name__ == "__main__":
    unittest.main()
