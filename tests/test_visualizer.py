import sys
import unittest
from unittest.mock import Mock, patch

from src.recognition.gesture_engine import (
    GestureDiagnostics,
    GestureState,
    HandPose,
)
from src.webcam.visualizer import FOOTER_TEXT, FrameState, Visualizer


def draw_texts(state: FrameState, **visualizer_options) -> list[str]:
    cv2 = Mock()
    with (
        patch.dict(sys.modules, {"cv2": cv2}),
        patch.object(Visualizer, "_draw_hand"),
    ):
        Visualizer(**visualizer_options)._draw(
            Mock(shape=(480, 640, 3)), state, 30.0
        )
    # Each text is drawn twice (outline, then fill); keep one of each.
    return list(dict.fromkeys(call.args[1] for call in cv2.putText.call_args_list))


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
                left_pinch_ratio=0.2,
                right_pinch_ratio=0.6,
                fist_confirmation_frames=2,
                pinch_enabled=True,
                two_finger_hand=True,
            ),
        )

        with patch.dict(sys.modules, {"cv2": cv2}):
            Visualizer(show_details=True)._draw(Mock(shape=(480, 640, 3)), state, 30.0)

        text_lines = [call.args[1] for call in cv2.putText.call_args_list]
        self.assertIn("Raw: PINCH_LEFT_CLICK", text_lines)
        self.assertIn("Filter: tips together confirm 2/3", text_lines)
        self.assertIn("Pinch gate: armed", text_lines)
        self.assertIn("Pose: NONE", text_lines)
        self.assertIn(
            "Flex 3D I/M/R/P: no depth (2D claw rules)", text_lines
        )
        self.assertIn("Index-only history pose: not detected", text_lines)
        self.assertIn("Two-finger tab pose: detected", text_lines)
        self.assertIn("Pointer: off (show claw pose)", text_lines)
        self.assertIn("Curl I/M/R/P: 0.80/1.20/--/1.10 (2/4)", text_lines)
        self.assertIn("Pinch L/R: 0.20/0.60 (threshold 0.45)", text_lines)

    def test_hud_reports_3d_flexion(self) -> None:
        cv2 = Mock()
        state = FrameState(
            [],
            GestureState.IDLE,
            None,
            None,
            GestureDiagnostics(finger_flex=(48.4, 52.0, 171.0, None)),
        )

        with patch.dict(sys.modules, {"cv2": cv2}):
            Visualizer(show_details=True)._draw(Mock(shape=(480, 640, 3)), state, 30.0)

        text_lines = [call.args[1] for call in cv2.putText.call_args_list]
        self.assertIn("Flex 3D I/M/R/P: 48/52/171/-- (claw 35-150)", text_lines)

    def test_hud_reports_pointer_state(self) -> None:
        for diagnostics, expected in (
            (GestureDiagnostics(pointer_arming_frames=2), "Pointer: arming 2/5"),
            (GestureDiagnostics(pointer_active=True), "Pointer: moving cursor"),
            (
                GestureDiagnostics(pointer_active=True, pointer_frozen=True),
                "Pointer: frozen for click",
            ),
        ):
            with self.subTest(expected=expected):
                cv2 = Mock()
                state = FrameState([], GestureState.IDLE, None, None, diagnostics)

                with patch.dict(sys.modules, {"cv2": cv2}):
                    Visualizer(show_details=True)._draw(Mock(shape=(480, 640, 3)), state, 30.0)

                text_lines = [
                    call.args[1] for call in cv2.putText.call_args_list
                ]
                self.assertIn(expected, text_lines)

    def test_compact_hud_shows_only_mode_hint_fps_and_keys(self) -> None:
        texts = draw_texts(
            FrameState(
                [Mock()],
                GestureState.IDLE,
                None,
                None,
                GestureDiagnostics(stable_pose=HandPose.TWO_FINGER),
            ),
            backend_name="MediaPipe",
        )

        self.assertEqual(
            texts,
            [
                "TABS",
                "swipe sideways: previous / next",
                "30 FPS",
                FOOTER_TEXT,
            ],
        )

    def test_compact_hud_labels_each_mode(self) -> None:
        hand = [Mock()]
        for state, expected in (
            (FrameState([], GestureState.IDLE, None, None), "NO HAND"),
            (FrameState(hand, GestureState.IDLE, None, None), "READY"),
            (
                FrameState(
                    hand, GestureState.IDLE, None, None,
                    GestureDiagnostics(pointer_arming_frames=2),
                ),
                "CURSOR",
            ),
            (
                FrameState(
                    hand, GestureState.IDLE, None, None,
                    GestureDiagnostics(stable_pose=HandPose.OPEN_PALM),
                ),
                "SCROLL",
            ),
            (
                FrameState(
                    hand, GestureState.IDLE, None, None,
                    GestureDiagnostics(stable_pose=HandPose.INDEX_ONLY),
                ),
                "HISTORY",
            ),
            (
                FrameState(
                    hand, GestureState.PINCH_RIGHT_CLICK, None, None,
                    GestureDiagnostics(pointer_active=True),
                ),
                "RIGHT CLICK",
            ),
        ):
            with self.subTest(expected=expected):
                self.assertEqual(draw_texts(state)[0], expected)

    def test_alert_is_shortened_to_the_action_name(self) -> None:
        texts = draw_texts(
            FrameState(
                [],
                GestureState.IDLE,
                None,
                "[WAVE COMMAND - TRANSLATOR] Gesture detected! -> "
                "Action: NEXT_TAB (Executed)",
            )
        )

        self.assertIn("NEXT TAB", texts)
        self.assertFalse(any("TRANSLATOR" in text for text in texts))

    def test_d_key_toggles_the_detailed_hud(self) -> None:
        cv2 = Mock()
        camera = cv2.VideoCapture.return_value
        camera.isOpened.return_value = True
        camera.read.return_value = (True, object())
        cv2.waitKey.side_effect = [ord("d"), -1, ord("d"), ord("q")]
        visualizer = Visualizer()
        modes = []

        with (
            patch.dict(sys.modules, {"cv2": cv2}),
            patch.object(
                Visualizer,
                "_draw",
                lambda self, *_args: modes.append(self.show_details),
            ),
        ):
            visualizer.run(
                lambda _frame: FrameState([], GestureState.IDLE, None, None)
            )

        self.assertEqual(modes, [False, True, True, False])

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
