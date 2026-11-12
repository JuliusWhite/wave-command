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


def make_partially_open_hand() -> HandPrediction:
    hand = make_curled_hand()
    points = list(hand.keypoints)
    points[6] = Keypoint(6, 100.0, 70.0, 1.0, 0.5, 0.5)
    points[8] = Keypoint(8, 100.0, 20.0, 1.0, 0.5, 0.5)
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


def make_pointing_hand() -> HandPrediction:
    hand = make_curled_hand()
    points = list(hand.keypoints)
    points[6] = Keypoint(6, 100.0, 70.0, 1.0, 0.5, 0.5)
    points[8] = Keypoint(8, 100.0, 20.0, 1.0, 0.5, 0.5)
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

    def test_index_pointing_pose_is_pointer_move(self) -> None:
        result = GestureEngine().evaluate(make_pointing_hand())

        self.assertEqual(result.gesture, GestureState.POINTER_MOVE)

    def test_open_distance_is_idle(self) -> None:
        result = GestureEngine(pinch_threshold=0.35).evaluate(make_hand(50, 150))
        self.assertEqual(result.gesture, GestureState.IDLE)

    def test_default_pinch_threshold_requires_a_closer_pinch(self) -> None:
        early = GestureEngine().evaluate(make_hand(100, 132))
        pinched = GestureEngine().evaluate(make_hand(100, 128))

        self.assertEqual(early.gesture, GestureState.IDLE)
        self.assertEqual(pinched.gesture, GestureState.PINCH_LEFT_CLICK)

    def test_low_confidence_fist_landmark_does_not_block_pinch(self) -> None:
        hand = make_hand(100, 120)
        points = list(hand.keypoints)
        points[6] = Keypoint(6, 100.0, 100.0, 0.1, 0.5, 0.5)

        result = GestureEngine().evaluate(
            HandPrediction(tuple(points), hand.confidence)
        )

        self.assertEqual(result.gesture, GestureState.PINCH_LEFT_CLICK)

    def test_curled_fingers_are_a_close_window_gesture(self) -> None:
        engine = GestureEngine()
        for _ in range(2):
            self.assertEqual(
                engine.evaluate(make_curled_hand()).gesture,
                GestureState.IDLE,
            )
        self.assertEqual(engine.diagnostics.fist_confirmation_frames, 2)
        result = engine.evaluate(make_curled_hand())

        self.assertEqual(result.gesture, GestureState.FIST_CLOSE_WINDOW)
        self.assertTrue(engine.diagnostics.fist_active)
        self.assertEqual(engine.diagnostics.fist_confirmation_frames, 0)

    def test_fist_recognizes_when_thumb_and_one_tip_are_occluded(self) -> None:
        engine = GestureEngine()
        for _ in range(3):
            result = engine.evaluate(
                make_curled_hand(thumb_confidence=0.1, low_confidence_tip=8)
            )

        self.assertEqual(result.gesture, GestureState.FIST_CLOSE_WINDOW)

    def test_brief_release_noise_does_not_retrigger_fist(self) -> None:
        engine = GestureEngine()
        closed_hand = make_curled_hand()
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

        closed_hand = make_curled_hand()
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
            process(make_curled_hand())
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
