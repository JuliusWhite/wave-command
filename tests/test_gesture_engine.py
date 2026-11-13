import unittest

from src.recognition.gesture_engine import GestureEngine, GestureState
from src.recognition.hand_detector import HandPrediction, Keypoint
from src.translation.command_translator import CommandTranslator


def make_hand(
    thumb_x: float,
    index_x: float,
    middle_x: float = 300.0,
) -> HandPrediction:
    points = [
        Keypoint(i, 100.0, 100.0, 1.0, 0.5, 0.5)
        for i in range(21)
    ]
    points[0] = Keypoint(0, 100.0, 100.0, 1.0, 0.5, 0.5)
    points[4] = Keypoint(4, thumb_x, 100.0, 1.0, 0.5, 0.5)
    points[8] = Keypoint(8, index_x, 100.0, 1.0, 0.5, 0.5)
    points[9] = Keypoint(9, 100.0, 200.0, 1.0, 0.5, 0.5)
    points[12] = Keypoint(12, middle_x, 100.0, 1.0, 0.5, 0.5)
    return HandPrediction(tuple(points), 1.0)


def make_open_hand(wrist_y: float, wrist_x: float = 100.0) -> HandPrediction:
    horizontal_offset = wrist_x - 100.0
    points = [
        Keypoint(i, 100.0 + horizontal_offset, wrist_y, 1.0, 0.5, 0.5)
        for i in range(21)
    ]
    points[0] = Keypoint(0, wrist_x, wrist_y, 1.0, 0.5, 0.5)
    points[4] = Keypoint(
        4, 20.0 + horizontal_offset, wrist_y - 10.0, 1.0, 0.5, 0.5
    )
    middle_mcp_y = wrist_y - 100.0
    for mcp_index, pip_index, tip_index, x in (
        (5, 6, 8, 60.0),
        (9, 10, 12, 100.0),
        (13, 14, 16, 140.0),
        (17, 18, 20, 180.0),
    ):
        points[mcp_index] = Keypoint(
            mcp_index, x + horizontal_offset, middle_mcp_y, 1.0, 0.5, 0.5
        )
        points[pip_index] = Keypoint(
            pip_index, x + horizontal_offset, middle_mcp_y - 30.0,
            1.0, 0.5, 0.5,
        )
        points[tip_index] = Keypoint(
            tip_index, x + horizontal_offset, middle_mcp_y - 70.0,
            1.0, 0.5, 0.5,
        )
    return HandPrediction(tuple(points), 1.0)


def make_two_finger_hand(
    wrist_y: float,
    wrist_x: float = 100.0,
) -> HandPrediction:
    hand = make_open_hand(wrist_y, wrist_x)
    points = list(hand.keypoints)
    for mcp_index, tip_index in ((13, 16), (17, 20)):
        mcp = points[mcp_index]
        points[tip_index] = Keypoint(
            tip_index, mcp.x, mcp.y - 10.0, 1.0, 0.5, 0.5
        )
    return HandPrediction(tuple(points), hand.confidence)


def make_two_finger_hand_with_right_pinch(
    wrist_y: float,
    wrist_x: float = 100.0,
) -> HandPrediction:
    hand = make_two_finger_hand(wrist_y, wrist_x)
    points = list(hand.keypoints)
    middle_tip = points[12]
    points[4] = Keypoint(
        4, middle_tip.x + 30.0, middle_tip.y, 1.0, 0.5, 0.5
    )
    return HandPrediction(tuple(points), hand.confidence)


def make_bent_two_finger_hand(
    wrist_y: float,
    wrist_x: float = 100.0,
) -> HandPrediction:
    hand = make_two_finger_hand(wrist_y, wrist_x)
    points = list(hand.keypoints)
    for pip_index, tip_index in ((14, 16), (18, 20)):
        pip = points[pip_index]
        points[tip_index] = Keypoint(
            tip_index, pip.x + 40.0, pip.y, 1.0, 0.5, 0.5
        )
    return HandPrediction(tuple(points), hand.confidence)


def move_wrist(
    hand: HandPrediction,
    wrist_y: float | None = None,
    wrist_x: float | None = None,
) -> HandPrediction:
    points = list(hand.keypoints)
    wrist = points[0]
    points[0] = Keypoint(
        0,
        wrist.x if wrist_x is None else wrist_x,
        wrist.y if wrist_y is None else wrist_y,
        wrist.conf,
        wrist.normalized_x,
        wrist.normalized_y,
    )
    return HandPrediction(tuple(points), hand.confidence)


def make_curled_hand(
    thumb_confidence: float = 1.0,
    low_confidence_tip: int | None = None,
) -> HandPrediction:
    hand = make_hand(100, 120)
    points = list(hand.keypoints)
    for pip_index, tip_index, pip_x in (
        (6, 8, 65),
        (10, 12, 80),
        (14, 16, 120),
        (18, 20, 135),
    ):
        points[pip_index] = Keypoint(pip_index, pip_x, 70.0, 1.0, 0.5, 0.5)
        points[tip_index] = Keypoint(tip_index, 100.0, 120.0, 1.0, 0.5, 0.5)

    thumb = points[4]
    points[4] = Keypoint(
        4, thumb.x, thumb.y, thumb_confidence, thumb.normalized_x,
        thumb.normalized_y,
    )
    if low_confidence_tip is not None:
        tip = points[low_confidence_tip]
        points[low_confidence_tip] = Keypoint(
            tip.id, tip.x, tip.y, 0.1, tip.normalized_x, tip.normalized_y
        )
    return HandPrediction(tuple(points), hand.confidence)


def make_tips_together_hand(
    thumb_confidence: float = 1.0,
    low_confidence_tip: int | None = None,
) -> HandPrediction:
    hand = make_hand(100, 120)
    points = list(hand.keypoints)
    points[4] = Keypoint(4, 100.0, 100.0, thumb_confidence, 0.5, 0.5)
    for tip_index in (8, 12, 16, 20):
        points[tip_index] = Keypoint(tip_index, 100.0, 100.0, 1.0, 0.5, 0.5)
    if low_confidence_tip is not None:
        tip = points[low_confidence_tip]
        points[low_confidence_tip] = Keypoint(
            tip.id, tip.x, tip.y, 0.1, tip.normalized_x, tip.normalized_y
        )
    return HandPrediction(tuple(points), hand.confidence)


def make_partially_open_hand() -> HandPrediction:
    hand = make_tips_together_hand()
    points = list(hand.keypoints)
    points[8] = Keypoint(8, 150.0, 100.0, 1.0, 0.5, 0.5)
    return HandPrediction(tuple(points), hand.confidence)


def make_left_pinch_with_curled_fingers() -> HandPrediction:
    hand = make_curled_hand()
    points = list(hand.keypoints)
    points[4] = Keypoint(4, 100.0, 20.0, 1.0, 0.5, 0.5)
    points[8] = Keypoint(8, 100.0, 20.0, 1.0, 0.5, 0.5)
    return HandPrediction(tuple(points), hand.confidence)


def make_opening_left_pinch() -> HandPrediction:
    hand = make_left_pinch_with_curled_fingers()
    points = list(hand.keypoints)
    points[10] = Keypoint(10, 100.0, 160.0, 1.0, 0.5, 0.5)
    points[12] = Keypoint(12, 100.0, 100.0, 1.0, 0.5, 0.5)
    return HandPrediction(tuple(points), hand.confidence)


class GestureEngineTests(unittest.TestCase):
    def test_thumb_index_pinch_is_left_click(self) -> None:
        result = GestureEngine(pinch_threshold=0.35).evaluate(make_hand(100, 120))
        self.assertEqual(result.gesture, GestureState.PINCH_LEFT_CLICK)
        self.assertAlmostEqual(result.pinch_distance_ratio, 0.2)

    def test_thumb_middle_pinch_is_right_click(self) -> None:
        result = GestureEngine(pinch_threshold=0.35).evaluate(
            make_hand(100, 250, middle_x=120)
        )
        self.assertEqual(result.gesture, GestureState.PINCH_RIGHT_CLICK)
        self.assertAlmostEqual(result.pinch_distance_ratio, 0.2)

    def test_left_pinch_wins_when_other_fingers_are_curled(self) -> None:
        result = GestureEngine().evaluate(make_left_pinch_with_curled_fingers())

        self.assertEqual(result.gesture, GestureState.PINCH_LEFT_CLICK)

    def test_open_distance_is_idle(self) -> None:
        result = GestureEngine(pinch_threshold=0.35).evaluate(make_hand(50, 150))
        self.assertEqual(result.gesture, GestureState.IDLE)

    def test_default_pinch_threshold_recognizes_both_pinches_sooner(self) -> None:
        left_pinch = GestureEngine().evaluate(make_hand(100, 142))
        right_pinch = GestureEngine().evaluate(
            make_hand(100, 250, middle_x=142)
        )
        open_hand = GestureEngine().evaluate(make_hand(100, 146))

        self.assertEqual(left_pinch.gesture, GestureState.PINCH_LEFT_CLICK)
        self.assertEqual(right_pinch.gesture, GestureState.PINCH_RIGHT_CLICK)
        self.assertEqual(open_hand.gesture, GestureState.IDLE)

    def test_similar_left_and_right_pinch_ratios_are_ambiguous(self) -> None:
        result = GestureEngine().evaluate(
            make_hand(100, 124, middle_x=128)
        )

        self.assertEqual(result.gesture, GestureState.IDLE)
        self.assertAlmostEqual(result.pinch_distance_ratio, 0.24)

    def test_clearly_dominant_right_pinch_wins_when_both_are_active(self) -> None:
        result = GestureEngine().evaluate(
            make_hand(100, 130, middle_x=120)
        )

        self.assertEqual(result.gesture, GestureState.PINCH_RIGHT_CLICK)

    def test_clearly_dominant_left_pinch_wins_when_both_are_active(self) -> None:
        result = GestureEngine().evaluate(
            make_hand(100, 120, middle_x=130)
        )

        self.assertEqual(result.gesture, GestureState.PINCH_LEFT_CLICK)

    @unittest.skip("Scroll gesture temporarily disabled")
    def test_open_hand_swing_bottom_to_top_scrolls_down(self) -> None:
        engine = GestureEngine()

        self.assertEqual(
            engine.evaluate(make_open_hand(300.0)).gesture,
            GestureState.IDLE,
        )
        result = engine.evaluate(
            move_wrist(make_open_hand(300.0), 220.0)
        )

        self.assertEqual(result.gesture, GestureState.SCROLL_DOWN)
        self.assertEqual(
            engine.evaluate(make_open_hand(220.0)).gesture,
            GestureState.IDLE,
        )
        self.assertEqual(
            engine.evaluate(make_open_hand(140.0)).gesture,
            GestureState.SCROLL_DOWN,
        )

    @unittest.skip("Scroll gesture temporarily disabled")
    def test_open_hand_swing_top_to_bottom_scrolls_up(self) -> None:
        engine = GestureEngine()

        engine.evaluate(make_open_hand(220.0))
        result = engine.evaluate(
            move_wrist(make_open_hand(220.0), 360.0)
        )

        self.assertEqual(result.gesture, GestureState.SCROLL_UP)

    @unittest.skip("Scroll gesture temporarily disabled")
    def test_scroll_waits_for_enough_open_hand_movement(self) -> None:
        engine = GestureEngine()

        engine.evaluate(make_open_hand(300.0))
        self.assertEqual(
            engine.evaluate(
                move_wrist(make_open_hand(300.0), 270.0)
            ).gesture,
            GestureState.IDLE,
        )
        self.assertEqual(
            engine.evaluate(
                move_wrist(make_open_hand(300.0), 220.0)
            ).gesture,
            GestureState.SCROLL_DOWN,
        )

    @unittest.skip("Scroll gesture temporarily disabled")
    def test_non_open_hand_resets_scroll_tracking(self) -> None:
        engine = GestureEngine()

        engine.evaluate(make_open_hand(300.0))
        engine.evaluate(make_hand(50.0, 150.0))
        self.assertEqual(
            engine.evaluate(make_open_hand(220.0)).gesture,
            GestureState.IDLE,
        )
        self.assertEqual(
            engine.evaluate(make_open_hand(140.0)).gesture,
            GestureState.SCROLL_DOWN,
        )

    def test_open_hand_swipe_right_to_left_selects_previous_tab(self) -> None:
        engine = GestureEngine()
        start = make_open_hand(300.0, wrist_x=200.0)

        engine.evaluate(start)
        result = engine.evaluate(make_open_hand(300.0, wrist_x=0.0))

        self.assertEqual(result.gesture, GestureState.PREVIOUS_TAB)
        self.assertEqual(
            engine.evaluate(make_open_hand(300.0, wrist_x=0.0)).gesture,
            GestureState.IDLE,
        )

    def test_open_hand_swipe_left_to_right_selects_next_tab(self) -> None:
        engine = GestureEngine()
        start = make_open_hand(300.0, wrist_x=0.0)

        engine.evaluate(start)
        result = engine.evaluate(make_open_hand(300.0, wrist_x=200.0))

        self.assertEqual(result.gesture, GestureState.NEXT_TAB)

    def test_two_finger_swipe_right_to_left_goes_back(self) -> None:
        engine = GestureEngine()

        engine.evaluate(make_two_finger_hand(300.0, wrist_x=200.0))
        result = engine.evaluate(
            make_two_finger_hand(300.0, wrist_x=0.0)
        )

        self.assertEqual(result.gesture, GestureState.BACK)

    def test_two_finger_swipe_left_to_right_goes_forward(self) -> None:
        engine = GestureEngine()

        engine.evaluate(make_two_finger_hand(300.0, wrist_x=0.0))
        result = engine.evaluate(
            make_two_finger_hand(300.0, wrist_x=200.0)
        )

        self.assertEqual(result.gesture, GestureState.FORWARD)

    def test_two_finger_swipe_does_not_require_thumb_visibility(self) -> None:
        engine = GestureEngine()
        start = make_two_finger_hand(300.0, wrist_x=200.0)
        start_points = list(start.keypoints)
        thumb = start_points[4]
        start_points[4] = Keypoint(
            4, thumb.x, thumb.y, 0.1, thumb.normalized_x, thumb.normalized_y
        )
        end = make_two_finger_hand(300.0, wrist_x=0.0)
        end_points = list(end.keypoints)
        thumb = end_points[4]
        end_points[4] = Keypoint(
            4, thumb.x, thumb.y, 0.1, thumb.normalized_x, thumb.normalized_y
        )

        engine.evaluate(HandPrediction(tuple(start_points), start.confidence))
        result = engine.evaluate(HandPrediction(tuple(end_points), end.confidence))

        self.assertEqual(result.gesture, GestureState.BACK)

    def test_two_finger_pose_suppresses_right_pinch_until_swipe(self) -> None:
        engine = GestureEngine()
        start = make_two_finger_hand_with_right_pinch(
            300.0,
            wrist_x=200.0,
        )
        end = make_two_finger_hand_with_right_pinch(
            300.0,
            wrist_x=0.0,
        )

        self.assertEqual(engine.evaluate(start).gesture, GestureState.IDLE)
        self.assertEqual(engine.evaluate(end).gesture, GestureState.BACK)

    def test_bent_ring_and_pinky_are_down_even_when_their_tips_are_far_from_mcp(
        self,
    ) -> None:
        engine = GestureEngine()

        engine.evaluate(make_bent_two_finger_hand(300.0, wrist_x=200.0))
        result = engine.evaluate(
            make_bent_two_finger_hand(300.0, wrist_x=0.0)
        )

        self.assertEqual(result.gesture, GestureState.BACK)

    def test_tab_swipe_requires_enough_horizontal_wrist_movement(self) -> None:
        engine = GestureEngine()
        start = make_open_hand(300.0, wrist_x=200.0)

        engine.evaluate(start)
        self.assertEqual(
            engine.evaluate(make_open_hand(300.0, wrist_x=100.0)).gesture,
            GestureState.IDLE,
        )
        self.assertEqual(
            engine.evaluate(make_open_hand(300.0, wrist_x=50.0)).gesture,
            GestureState.PREVIOUS_TAB,
        )

    def test_vertical_open_hand_movement_does_not_switch_tabs(self) -> None:
        engine = GestureEngine()
        start = make_open_hand(300.0)

        engine.evaluate(start)
        result = engine.evaluate(move_wrist(start, wrist_y=220.0))

        self.assertEqual(result.gesture, GestureState.IDLE)

    def test_pinch_is_not_misclassified_as_an_open_hand_swipe(self) -> None:
        hand = make_open_hand(300.0)
        points = list(hand.keypoints)
        index_tip = points[8]
        points[4] = Keypoint(
            4,
            index_tip.x,
            index_tip.y,
            1.0,
            index_tip.normalized_x,
            index_tip.normalized_y,
        )

        result = GestureEngine().evaluate(
            HandPrediction(tuple(points), hand.confidence)
        )

        self.assertEqual(result.gesture, GestureState.PINCH_LEFT_CLICK)

    def test_low_confidence_fist_landmark_does_not_block_pinch(self) -> None:
        hand = make_hand(100, 120)
        points = list(hand.keypoints)
        points[6] = Keypoint(6, 100.0, 100.0, 0.1, 0.5, 0.5)

        result = GestureEngine().evaluate(
            HandPrediction(tuple(points), hand.confidence)
        )

        self.assertEqual(result.gesture, GestureState.PINCH_LEFT_CLICK)

    @unittest.skip("Close gesture temporarily disabled")
    def test_all_fingertips_together_is_a_close_window_gesture(self) -> None:
        engine = GestureEngine()
        for _ in range(2):
            self.assertEqual(
                engine.evaluate(make_tips_together_hand()).gesture,
                GestureState.IDLE,
            )
        self.assertEqual(engine.diagnostics.fist_confirmation_frames, 2)
        result = engine.evaluate(make_tips_together_hand())

        self.assertEqual(result.gesture, GestureState.FIST_CLOSE_WINDOW)
        self.assertTrue(engine.diagnostics.fist_active)
        self.assertEqual(engine.diagnostics.fist_confirmation_frames, 0)

    def test_curled_fingers_without_together_tips_do_not_close_window(self) -> None:
        engine = GestureEngine()
        for _ in range(3):
            result = engine.evaluate(make_curled_hand())

        self.assertEqual(result.gesture, GestureState.IDLE)

    @unittest.skip("Close gesture temporarily disabled")
    def test_close_requires_all_fingertips_to_be_visible(self) -> None:
        engine = GestureEngine()
        for _ in range(3):
            result = engine.evaluate(
                make_tips_together_hand(
                    thumb_confidence=0.1,
                    low_confidence_tip=8,
                )
            )

        self.assertEqual(result.gesture, GestureState.IDLE)

    @unittest.skip("Close gesture temporarily disabled")
    def test_brief_release_noise_does_not_retrigger_fist(self) -> None:
        engine = GestureEngine()
        closed_hand = make_tips_together_hand()
        open_hand = make_hand(50, 150)
        close_calls: list[bool] = []
        translator = CommandTranslator(
            cooldown_sec=0,
            close_window=lambda: close_calls.append(True),
        )

        def process(hand: HandPrediction) -> GestureState:
            gesture = engine.evaluate(hand).gesture
            translator.process(gesture)
            return gesture

        for _ in range(3):
            gesture = process(closed_hand)
        self.assertEqual(gesture, GestureState.FIST_CLOSE_WINDOW)

        self.assertEqual(
            process(open_hand),
            GestureState.FIST_CLOSE_WINDOW,
        )
        self.assertEqual(
            process(closed_hand),
            GestureState.FIST_CLOSE_WINDOW,
        )

        for _ in range(5):
            self.assertEqual(process(open_hand), GestureState.FIST_CLOSE_WINDOW)
        self.assertEqual(process(open_hand), GestureState.IDLE)

        for _ in range(3):
            process(open_hand)
        self.assertEqual(process(closed_hand), GestureState.IDLE)
        self.assertEqual(process(closed_hand), GestureState.IDLE)
        self.assertEqual(
            process(closed_hand),
            GestureState.FIST_CLOSE_WINDOW,
        )
        self.assertEqual(close_calls, [True, True])

    @unittest.skip("Close gesture temporarily disabled")
    def test_partially_opening_fist_does_not_trigger_close_again(self) -> None:
        engine = GestureEngine()
        close_calls: list[bool] = []
        translator = CommandTranslator(
            cooldown_sec=0,
            close_window=lambda: close_calls.append(True),
        )

        def process(hand: HandPrediction) -> GestureState:
            gesture = engine.evaluate(hand).gesture
            translator.process(gesture)
            return gesture

        closed_hand = make_tips_together_hand()
        open_hand = make_hand(50, 150)
        partial_hand = make_partially_open_hand()

        for _ in range(3):
            process(closed_hand)
        self.assertEqual(close_calls, [True])

        for _ in range(6):
            process(open_hand)
        for _ in range(10):
            self.assertNotEqual(
                process(partial_hand),
                GestureState.FIST_CLOSE_WINDOW,
            )
        self.assertEqual(close_calls, [True])

        for _ in range(3):
            process(open_hand)
        for _ in range(3):
            process(closed_hand)
        self.assertEqual(close_calls, [True, True])

    @unittest.skip("Close gesture temporarily disabled")
    def test_opening_fist_does_not_trigger_a_click(self) -> None:
        engine = GestureEngine()
        close_calls: list[bool] = []
        clicks: list[str] = []
        translator = CommandTranslator(
            cooldown_sec=0,
            click=clicks.append,
            close_window=lambda: close_calls.append(True),
        )

        def process(hand: HandPrediction) -> GestureState:
            gesture = engine.evaluate(hand).gesture
            translator.process(gesture)
            return gesture

        for _ in range(3):
            process(make_tips_together_hand())
        self.assertEqual(close_calls, [True])

        for _ in range(6):
            process(make_left_pinch_with_curled_fingers())
        for _ in range(6):
            process(make_opening_left_pinch())
        self.assertEqual(clicks, [])

        for _ in range(3):
            process(make_hand(50, 150))
        process(make_hand(100, 120))

        self.assertEqual(clicks, ["left"])
        self.assertEqual(close_calls, [True])

    def test_missing_hand_is_idle(self) -> None:
        result = GestureEngine().evaluate(None)
        self.assertEqual(result.gesture, GestureState.IDLE)
        self.assertIsNone(result.pinch_distance_ratio)


if __name__ == "__main__":
    unittest.main()
