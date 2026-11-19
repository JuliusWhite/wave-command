import sys
import unittest
from unittest.mock import Mock, patch

from src.recognition.gesture_engine import (
    GestureDiagnostics,
    GestureState,
    HandPose,
)
from src.webcam.overlay import bgr_to_hex
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


class FakeOverlay:
    """Stands in for StatusOverlay; quits after a set number of frames."""

    def __init__(
        self,
        frames: int,
        toggle_on: set[int] = frozenset(),
        bypass_on: set[int] = frozenset(),
    ) -> None:
        self.frames = frames
        self.toggle_on = toggle_on
        self.bypass_on = bypass_on
        self.statuses = []
        self.quit_requested = False
        self.closed = False

    def consume_camera_toggle(self) -> bool:
        return len(self.statuses) in self.toggle_on

    def consume_bypass_toggle(self) -> bool:
        return len(self.statuses) in self.bypass_on

    def show(self, status) -> None:
        self.statuses.append(status)

    def pump(self) -> None:
        self.quit_requested = len(self.statuses) >= self.frames

    def close(self) -> None:
        self.closed = True


def run_with_overlay(overlay, states, show_camera=False, keys=None):
    cv2 = Mock()
    camera = cv2.VideoCapture.return_value
    camera.isOpened.return_value = True
    camera.read.return_value = (True, object())
    cv2.waitKey.side_effect = keys or (lambda _delay: -1)
    cv2.getWindowProperty.return_value = 1.0
    states = list(states)
    with (
        patch.dict(sys.modules, {"cv2": cv2}),
        patch.object(Visualizer, "_draw"),
    ):
        Visualizer(overlay=overlay, show_camera=show_camera).run(
            lambda _frame: states.pop(0) if len(states) > 1 else states[0]
        )
    return cv2


class OverlayTests(unittest.TestCase):
    def test_bgr_colors_become_tk_hex(self) -> None:
        self.assertEqual(bgr_to_hex((40, 100, 255)), "#ff6428")

    def test_overlay_reports_hand_presence_mode_and_action(self) -> None:
        visualizer = Visualizer(overlay=FakeOverlay(1), show_camera=False)

        no_hand = visualizer.overlay_status(
            FrameState([], GestureState.IDLE, None, None)
        )
        scrolling = visualizer.overlay_status(
            FrameState(
                [Mock()],
                GestureState.IDLE,
                None,
                "[WAVE COMMAND - TRANSLATOR] Gesture detected! -> "
                "Action: NEXT_TAB (Executed)",
                GestureDiagnostics(stable_pose=HandPose.OPEN_PALM),
            )
        )

        self.assertEqual(no_hand.label, "NO HAND")
        self.assertEqual(no_hand.hand_text, "No hand in view")
        self.assertEqual(no_hand.hand_color, Visualizer._OPEN_COLOR)
        self.assertIsNone(no_hand.action)
        self.assertFalse(no_hand.camera_visible)
        self.assertEqual(scrolling.label, "SCROLL")
        self.assertEqual(scrolling.color, Visualizer._SCROLL_COLOR)
        self.assertEqual(scrolling.hand_text, "Hand detected")
        self.assertEqual(scrolling.hand_color, Visualizer._PINCH_COLOR)
        self.assertEqual(scrolling.action, "NEXT TAB")

    def test_overlay_runs_without_the_camera_preview(self) -> None:
        overlay = FakeOverlay(3)

        cv2 = run_with_overlay(
            overlay, [FrameState([], GestureState.IDLE, None, None)]
        )

        self.assertEqual(len(overlay.statuses), 3)
        cv2.imshow.assert_not_called()
        self.assertTrue(overlay.closed)

    def test_camera_button_toggles_the_preview(self) -> None:
        overlay = FakeOverlay(5, toggle_on={1, 3})

        cv2 = run_with_overlay(
            overlay, [FrameState([], GestureState.IDLE, None, None)]
        )

        self.assertEqual(
            [status.camera_visible for status in overlay.statuses],
            [False, True, True, False, False],
        )
        self.assertEqual(cv2.imshow.call_count, 2)
        cv2.destroyWindow.assert_called_once()

    def test_c_key_hides_the_preview_but_keeps_the_overlay(self) -> None:
        overlay = FakeOverlay(3)

        cv2 = run_with_overlay(
            overlay,
            [FrameState([], GestureState.IDLE, None, None)],
            show_camera=True,
            keys=[ord("c")],
        )

        self.assertEqual(cv2.imshow.call_count, 1)
        self.assertEqual(len(overlay.statuses), 3)
        self.assertFalse(overlay.statuses[-1].camera_visible)

    def test_bypass_button_toggles_gesture_actions(self) -> None:
        overlay = FakeOverlay(4, bypass_on={1, 3})

        run_with_overlay(
            overlay, [FrameState([], GestureState.IDLE, None, None)]
        )

        self.assertEqual(
            [status.bypassed for status in overlay.statuses],
            [False, True, True, False],
        )

    def test_bypassed_status_overrides_the_mode(self) -> None:
        visualizer = Visualizer(
            overlay=FakeOverlay(1), show_camera=False, bypassed=True
        )

        status = visualizer.overlay_status(
            FrameState(
                [Mock()],
                GestureState.IDLE,
                None,
                None,
                GestureDiagnostics(stable_pose=HandPose.OPEN_PALM),
            )
        )

        self.assertEqual(status.label, "BYPASSED")
        self.assertIn("hand seen", status.hint)
        self.assertTrue(status.bypassed)

    def test_b_key_toggles_bypass(self) -> None:
        overlay = FakeOverlay(2)

        run_with_overlay(
            overlay,
            [FrameState([], GestureState.IDLE, None, None)],
            show_camera=True,
            keys=[ord("b"), -1],
        )

        self.assertEqual(
            [status.bypassed for status in overlay.statuses], [False, True]
        )

    def test_without_overlay_the_preview_is_always_shown(self) -> None:
        self.assertTrue(Visualizer(show_camera=False).show_camera)


if __name__ == "__main__":
    unittest.main()
