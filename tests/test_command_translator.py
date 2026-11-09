import unittest

from src.recognition.gesture_engine import GestureState
from src.translation.command_translator import CommandTranslator


class CommandTranslatorTests(unittest.TestCase):
    def test_emits_once_for_continuous_pinch_and_rearms_after_idle(self) -> None:
        messages: list[str] = []
        translator = CommandTranslator(cooldown_sec=0, emit=messages.append)

        first = translator.process(GestureState.PINCH_CLOSE_WINDOW)
        held = translator.process(GestureState.PINCH_CLOSE_WINDOW)
        translator.process(GestureState.IDLE)
        second = translator.process(GestureState.PINCH_CLOSE_WINDOW)

        self.assertIsNotNone(first)
        self.assertIsNone(held)
        self.assertIsNotNone(second)
        self.assertEqual(len(messages), 2)
        self.assertIn("CLOSE_WINDOW (Simulated)", messages[0])

    def test_cooldown_suppresses_new_pinch_transition(self) -> None:
        messages: list[str] = []
        current_time = [10.0]
        translator = CommandTranslator(
            cooldown_sec=1.5,
            clock=lambda: current_time[0],
            emit=messages.append,
        )

        translator.process(GestureState.PINCH_CLOSE_WINDOW)
        translator.process(GestureState.IDLE)
        current_time[0] = 11.0
        self.assertIsNone(translator.process(GestureState.PINCH_CLOSE_WINDOW))
        self.assertEqual(len(messages), 1)


if __name__ == "__main__":
    unittest.main()

