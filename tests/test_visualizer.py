import sys
import unittest
from unittest.mock import Mock, patch

from src.recognition.gesture_engine import GestureDiagnostics, GestureState
from src.webcam.visualizer import FrameState, Visualizer


class VisualizerTests(unittest.TestCase):
    def test_hud_displays_raw_pose_diagnostics(self) -> None:
        cv2 = Mock()
        state = FrameState(
            [],
            GestureState.IDLE,
            0.2,
            None,
            GestureDiagnostics(
                raw_gesture=GestureState.PINCH_LEFT_CLICK,
                curled_fingers=2,
                visible_fingers=4,
                curl_ratios=(0.8, 1.2, None, 1.1),
                thumb_curl_ratio=0.5,
                left_pinch_ratio=0.2,
                right_pinch_ratio=0.6,
                fist_confirmation_frames=2,
            ),
        )

        with patch.dict(sys.modules, {"cv2": cv2}):
            Visualizer()._draw(Mock(shape=(480, 640, 3)), state, 30.0)

        text_lines = [call.args[1] for call in cv2.putText.call_args_list]
        self.assertIn("Raw: PINCH_LEFT_CLICK", text_lines)
        self.assertIn("Filter: fist confirm 2/3", text_lines)
        self.assertIn("Curl I/M/R/P: 0.80/1.20/--/1.10 (2/4)", text_lines)
        self.assertIn("Thumb curl: 0.50", text_lines)
        self.assertIn("Pinch L/R: 0.20/0.60 (threshold 0.30)", text_lines)

    def test_fist_does_not_stop_the_webcam_loop(self) -> None:
        cv2 = Mock()
        camera = cv2.VideoCapture.return_value
        camera.isOpened.return_value = True
        camera.read.side_effect = [(True, object()), (True, object())]
        cv2.waitKey.side_effect = [-1, ord("q")]
        states = [
            FrameState([], GestureState.FIST_CLOSE_WINDOW, None, None),
            FrameState([], GestureState.IDLE, None, None),
        ]

        with (
            patch.dict(sys.modules, {"cv2": cv2}),
            patch.object(Visualizer, "_draw"),
        ):
            Visualizer().run(lambda _frame: states.pop(0))

        self.assertEqual(camera.read.call_count, 2)
        camera.release.assert_called_once()
        cv2.destroyAllWindows.assert_called_once()


if __name__ == "__main__":
    unittest.main()
