import unittest
from unittest.mock import Mock, patch

from src.recognition.gesture_engine import GestureResult, GestureState


class MainPipelineTests(unittest.TestCase):
    def test_moves_pointer_only_while_pointing_and_resets_afterward(self) -> None:
        import main

        hand = Mock()
        keypoints = [Mock() for _ in range(9)]
        keypoints[8].normalized_x = 0.25
        keypoints[8].normalized_y = 0.75
        hand.keypoints = keypoints
        frame = object()
        results = []

        with (
            patch("main.HandDetector") as detector_class,
            patch("main.GestureEngine") as gesture_engine_class,
            patch("main.CommandTranslator"),
            patch("main.PointerController") as pointer_controller_class,
            patch("main.Visualizer") as visualizer_class,
        ):
            detector = detector_class.return_value
            detector.detect.side_effect = [[hand], []]
            gesture_engine = gesture_engine_class.return_value
            gesture_engine.evaluate.side_effect = [
                GestureResult(GestureState.POINTER_MOVE, None),
                GestureResult(GestureState.IDLE, None),
            ]
            visualizer_class.return_value.run.side_effect = (
                lambda process_frame: results.extend(
                    (process_frame(frame), process_frame(frame))
                )
            )

            main.main()

        pointer_controller = pointer_controller_class.return_value
        pointer_controller.move_to.assert_called_once_with(0.25, 0.75)
        pointer_controller.reset.assert_called_once_with()
        self.assertEqual(len(results), 2)
        detector.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
