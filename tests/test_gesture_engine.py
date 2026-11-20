import unittest
from dataclasses import replace

from src.config import (
    POINTER_CLAW_MIN_FLEX_DEGREES,
    PINCH_ARM_TIMEOUT_SEC,
    SWIPE_RETURN_LOCKOUT_SEC,
    PINCH_CONFIRM_FRAMES,
    POSE_RELEASE_FRAMES,
    POINTER_ENTER_FRAMES,
    POINTER_EXIT_FRAMES,
    SCROLL_MAX_HAND_SIZES_PER_SEC,
    SCROLL_UNITS_PER_HAND_SIZE_PER_SEC,
)
from src.recognition.gesture_engine import GestureEngine, GestureState, HandPose
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


def rotate_hand(hand: HandPrediction, degrees: float) -> HandPrediction:
    import math

    wrist = hand.keypoints[0]
    radians = math.radians(degrees)
    points = []
    for point in hand.keypoints:
        dx = point.x - wrist.x
        dy = point.y - wrist.y
        points.append(
            Keypoint(
                point.id,
                wrist.x + dx * math.cos(radians) - dy * math.sin(radians),
                wrist.y + dx * math.sin(radians) + dy * math.cos(radians),
                point.conf,
                point.normalized_x,
                point.normalized_y,
            )
        )
    return HandPrediction(tuple(points), hand.confidence)


def make_narrow_open_hand() -> HandPrediction:
    hand = make_open_hand(300.0)
    points = list(hand.keypoints)
    for point_index, x in (
        (5, 95.0), (6, 95.0), (8, 95.0),
        (13, 101.0), (14, 101.0), (16, 101.0),
        (17, 105.0), (18, 105.0), (20, 105.0),
    ):
        point = points[point_index]
        points[point_index] = Keypoint(
            point.id, x, point.y, point.conf,
            point.normalized_x, point.normalized_y,
        )
    return HandPrediction(tuple(points), hand.confidence)


def make_index_only_hand(
    wrist_y: float,
    wrist_x: float = 100.0,
) -> HandPrediction:
    hand = make_open_hand(wrist_y, wrist_x)
    points = list(hand.keypoints)
    for mcp_index, tip_index in ((9, 12), (13, 16), (17, 20)):
        mcp = points[mcp_index]
        points[tip_index] = Keypoint(
            tip_index, mcp.x, mcp.y - 10.0, 1.0, 0.5, 0.5
        )
    return HandPrediction(tuple(points), hand.confidence)


def make_index_middle_hand(
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


def make_three_finger_hand(
    wrist_y: float,
    wrist_x: float = 100.0,
) -> HandPrediction:
    """Index, middle, and ring extended, pinky curled."""
    hand = make_open_hand(wrist_y, wrist_x)
    points = list(hand.keypoints)
    mcp = points[17]
    points[20] = Keypoint(20, mcp.x, mcp.y - 10.0, 1.0, 0.5, 0.5)
    return HandPrediction(tuple(points), hand.confidence)


def make_pointer_hand(
    wrist_y: float = 300.0,
    wrist_x: float = 100.0,
) -> HandPrediction:
    """Pinch-ready claw: ring and pinky curled, index and middle bent ~135°."""
    hand = make_open_hand(wrist_y, wrist_x)
    points = list(hand.keypoints)
    for mcp_index, tip_index in ((13, 16), (17, 20)):
        mcp = points[mcp_index]
        points[tip_index] = Keypoint(
            tip_index, mcp.x, mcp.y - 10.0, 1.0, 0.5, 0.5
        )
    for pip_index, tip_index in ((6, 8), (10, 12)):
        pip = points[pip_index]
        points[tip_index] = Keypoint(
            tip_index, pip.x + 25.0, pip.y - 25.0, 1.0, 0.5, 0.5
        )
    return HandPrediction(tuple(points), hand.confidence)


def with_thumb_at(hand: HandPrediction, x: float, y: float) -> HandPrediction:
    points = list(hand.keypoints)
    points[4] = Keypoint(4, x, y, 1.0, 0.5, 0.5)
    return HandPrediction(tuple(points), hand.confidence)


def start_pointer(engine: GestureEngine, wrist_x: float = 100.0) -> None:
    for _ in range(POINTER_ENTER_FRAMES):
        engine.evaluate(make_pointer_hand(wrist_x=wrist_x))


def confirm_pinch(engine: GestureEngine, hand: HandPrediction):
    """Hold a pinch for the confirmation frames; return the clicking frame."""
    for _ in range(PINCH_CONFIRM_FRAMES - 1):
        result = engine.evaluate(hand)
        assert result.gesture == GestureState.IDLE, result.gesture
    return engine.evaluate(hand)


def make_index_only_hand_with_right_pinch(
    wrist_y: float,
    wrist_x: float = 100.0,
) -> HandPrediction:
    hand = make_index_only_hand(wrist_y, wrist_x)
    points = list(hand.keypoints)
    middle_tip = points[12]
    points[4] = Keypoint(
        4, middle_tip.x + 30.0, middle_tip.y, 1.0, 0.5, 0.5
    )
    return HandPrediction(tuple(points), hand.confidence)


def make_bent_index_only_hand(
    wrist_y: float,
    wrist_x: float = 100.0,
) -> HandPrediction:
    hand = make_index_only_hand(wrist_y, wrist_x)
    points = list(hand.keypoints)
    for pip_index, tip_index in ((10, 12), (14, 16), (18, 20)):
        pip = points[pip_index]
        points[tip_index] = Keypoint(
            tip_index, pip.x + 40.0, pip.y, 1.0, 0.5, 0.5
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
    def test_each_pinch_requires_open_hand_and_hand_loss_resets_gate(self) -> None:
        engine = GestureEngine()
        left_pinch = make_hand(100, 120)
        right_pinch = make_hand(100, 250, middle_x=120)

        self.assertEqual(engine.evaluate(left_pinch).gesture, GestureState.IDLE)
        self.assertFalse(engine.diagnostics.pinch_enabled)

        self.assertEqual(
            engine.evaluate(make_open_hand(300.0)).gesture,
            GestureState.IDLE,
        )
        self.assertTrue(engine.diagnostics.pinch_enabled)
        self.assertEqual(
            confirm_pinch(engine, left_pinch).gesture,
            GestureState.PINCH_LEFT_CLICK,
        )
        self.assertFalse(engine.diagnostics.pinch_enabled)
        self.assertEqual(engine.evaluate(right_pinch).gesture, GestureState.IDLE)

        engine.evaluate(make_open_hand(300.0))
        self.assertEqual(
            confirm_pinch(engine, right_pinch).gesture,
            GestureState.PINCH_RIGHT_CLICK,
        )
        self.assertFalse(engine.diagnostics.pinch_enabled)

        engine.evaluate(None)
        self.assertFalse(engine.diagnostics.pinch_enabled)
        self.assertEqual(engine.evaluate(left_pinch).gesture, GestureState.IDLE)

    def test_pinch_gate_requires_upright_palm_facing_camera(self) -> None:
        for pose_name, open_pose in (
            ("rotated", rotate_hand(make_open_hand(300.0), 90.0)),
            ("narrow", make_narrow_open_hand()),
        ):
            with self.subTest(pose=pose_name):
                engine = GestureEngine()
                engine.evaluate(open_pose)

                self.assertFalse(engine.diagnostics.pinch_enabled)
                self.assertEqual(
                    engine.evaluate(make_hand(100, 120)).gesture,
                    GestureState.IDLE,
                )

    def test_thumb_index_pinch_is_left_click(self) -> None:
        engine = GestureEngine(pinch_threshold=0.35)
        engine.evaluate(make_open_hand(300.0))
        result = confirm_pinch(engine, make_hand(100, 120))
        self.assertEqual(result.gesture, GestureState.PINCH_LEFT_CLICK)
        self.assertAlmostEqual(result.pinch_distance_ratio, 0.2)

    def test_thumb_middle_pinch_is_right_click(self) -> None:
        engine = GestureEngine(pinch_threshold=0.35)
        engine.evaluate(make_open_hand(300.0))
        result = confirm_pinch(engine, make_hand(100, 250, middle_x=120))
        self.assertEqual(result.gesture, GestureState.PINCH_RIGHT_CLICK)
        self.assertAlmostEqual(result.pinch_distance_ratio, 0.2)

    def test_left_pinch_wins_when_other_fingers_are_curled(self) -> None:
        engine = GestureEngine()
        engine.evaluate(make_open_hand(300.0))
        result = confirm_pinch(engine, make_left_pinch_with_curled_fingers())

        self.assertEqual(result.gesture, GestureState.PINCH_LEFT_CLICK)

    def test_open_distance_is_idle(self) -> None:
        result = GestureEngine(pinch_threshold=0.35).evaluate(make_hand(50, 150))
        self.assertEqual(result.gesture, GestureState.IDLE)

    def test_default_pinch_threshold_recognizes_both_pinches_sooner(self) -> None:
        left_engine = GestureEngine()
        left_engine.evaluate(make_open_hand(300.0))
        left_pinch = confirm_pinch(left_engine, make_hand(100, 142))
        right_engine = GestureEngine()
        right_engine.evaluate(make_open_hand(300.0))
        right_pinch = confirm_pinch(
            right_engine, make_hand(100, 250, middle_x=142)
        )
        open_hand = GestureEngine().evaluate(make_hand(100, 146))

        self.assertEqual(left_pinch.gesture, GestureState.PINCH_LEFT_CLICK)
        self.assertEqual(right_pinch.gesture, GestureState.PINCH_RIGHT_CLICK)
        self.assertEqual(open_hand.gesture, GestureState.IDLE)

    def test_similar_left_and_right_pinch_ratios_are_ambiguous(self) -> None:
        engine = GestureEngine()
        engine.evaluate(make_open_hand(300.0))
        result = engine.evaluate(make_hand(100, 124, middle_x=128))

        self.assertEqual(result.gesture, GestureState.IDLE)
        self.assertAlmostEqual(result.pinch_distance_ratio, 0.24)

    def test_clearly_dominant_right_pinch_wins_when_both_are_active(self) -> None:
        engine = GestureEngine()
        engine.evaluate(make_open_hand(300.0))
        result = confirm_pinch(engine, make_hand(100, 130, middle_x=120))

        self.assertEqual(result.gesture, GestureState.PINCH_RIGHT_CLICK)

    def test_clearly_dominant_left_pinch_wins_when_both_are_active(self) -> None:
        engine = GestureEngine()
        engine.evaluate(make_open_hand(300.0))
        result = confirm_pinch(engine, make_hand(100, 120, middle_x=130))

        self.assertEqual(result.gesture, GestureState.PINCH_LEFT_CLICK)

    def test_open_hand_scroll_emits_smoothed_wheel_deltas_and_stops(self) -> None:
        current_time = [0.0]
        engine = GestureEngine(clock=lambda: current_time[0])

        engine.evaluate(make_open_hand(300.0))
        current_time[0] += 0.05
        initial_movement = engine.evaluate(make_open_hand(290.0))
        self.assertEqual(initial_movement.gesture, GestureState.IDLE)
        self.assertEqual(initial_movement.scroll_amount, 0)
        self.assertTrue(engine.diagnostics.pinch_enabled)

        scroll_results = []
        for wrist_y in (280.0, 270.0, 260.0):
            current_time[0] += 0.05
            scroll_results.append(
                engine.evaluate(make_open_hand(wrist_y))
            )

        self.assertTrue(
            all(
                result.gesture == GestureState.SCROLL_DOWN
                and result.scroll_amount < 0
                and abs(result.scroll_amount)
                <= (
                    SCROLL_MAX_HAND_SIZES_PER_SEC
                    * SCROLL_UNITS_PER_HAND_SIZE_PER_SEC
                    * 0.05
                    + 1
                )
                for result in scroll_results
            )
        )

        stationary_results = []
        for _ in range(12):
            current_time[0] += 0.05
            stationary_results.append(
                engine.evaluate(make_open_hand(260.0))
            )

        self.assertEqual(
            stationary_results[-1].gesture,
            GestureState.IDLE,
        )
        self.assertEqual(stationary_results[-1].scroll_amount, 0)

    def test_downward_open_hand_motion_emits_positive_scroll(self) -> None:
        current_time = [0.0]
        engine = GestureEngine(clock=lambda: current_time[0])

        engine.evaluate(make_open_hand(220.0))
        current_time[0] = 0.05
        first_movement = engine.evaluate(make_open_hand(230.0))
        self.assertEqual(first_movement.gesture, GestureState.IDLE)
        self.assertEqual(first_movement.scroll_amount, 0)

        current_time[0] = 0.1
        result = engine.evaluate(make_open_hand(240.0))

        self.assertEqual(result.gesture, GestureState.SCROLL_UP)
        self.assertGreater(result.scroll_amount, 0)

    def test_open_hand_scroll_ignores_sub_dead_zone_movement(self) -> None:
        current_time = [0.0]
        engine = GestureEngine(clock=lambda: current_time[0])

        engine.evaluate(make_open_hand(300.0))
        current_time[0] = 0.05
        result = engine.evaluate(make_open_hand(299.0))

        self.assertEqual(result.gesture, GestureState.IDLE)
        self.assertEqual(result.scroll_amount, 0)
        self.assertTrue(engine.diagnostics.pinch_enabled)

    def test_small_open_hand_jitter_does_not_scroll_or_disarm_pinches(
        self,
    ) -> None:
        current_time = [0.0]
        engine = GestureEngine(clock=lambda: current_time[0])
        engine.evaluate(make_open_hand(300.0))

        for wrist_y in (298.0, 302.0, 299.0, 301.0, 300.0):
            current_time[0] += 1 / 30
            result = engine.evaluate(make_open_hand(wrist_y))

            self.assertEqual(result.gesture, GestureState.IDLE)
            self.assertEqual(result.scroll_amount, 0)
            self.assertTrue(engine.diagnostics.pinch_enabled)

    def test_non_open_hand_resets_scroll_velocity(self) -> None:
        current_time = [0.0]
        engine = GestureEngine(clock=lambda: current_time[0])

        engine.evaluate(make_open_hand(300.0))
        for _ in range(POSE_RELEASE_FRAMES):
            engine.evaluate(make_hand(50.0, 150.0))
        self.assertEqual(engine.diagnostics.stable_pose, HandPose.NONE)
        current_time[0] = 0.05
        self.assertEqual(
            engine.evaluate(make_open_hand(220.0)).gesture,
            GestureState.IDLE,
        )
        current_time[0] = 0.1
        self.assertEqual(
            engine.evaluate(make_open_hand(210.0)).gesture,
            GestureState.IDLE,
        )
        current_time[0] = 0.15
        result = engine.evaluate(make_open_hand(200.0))

        self.assertEqual(result.gesture, GestureState.SCROLL_DOWN)
        self.assertLess(result.scroll_amount, 0)

    def test_horizontal_open_hand_swipes_do_not_change_tabs(self) -> None:
        for start_x, end_x in ((200.0, 0.0), (0.0, 200.0)):
            with self.subTest(start_x=start_x, end_x=end_x):
                engine = GestureEngine()
                engine.evaluate(make_open_hand(300.0, wrist_x=start_x))

                result = engine.evaluate(
                    make_open_hand(300.0, wrist_x=end_x)
                )

                self.assertEqual(result.gesture, GestureState.IDLE)
                self.assertEqual(result.scroll_amount, 0)

    def test_two_finger_swipes_change_tabs(self) -> None:
        for start_x, end_x, expected in (
            (200.0, 0.0, GestureState.PREVIOUS_TAB),
            (0.0, 200.0, GestureState.NEXT_TAB),
        ):
            with self.subTest(start_x=start_x, end_x=end_x):
                engine = GestureEngine()
                engine.evaluate(make_index_middle_hand(300.0, wrist_x=start_x))

                results = []
                step = (end_x - start_x) / 5
                for frame in range(1, 6):
                    results.append(
                        engine.evaluate(
                            make_index_middle_hand(
                                300.0,
                                wrist_x=start_x + step * frame,
                            )
                        )
                    )

                # 0.4 hand sizes per frame: the switch fires once on the
                # fourth frame (1.6 >= 1.25), then waits for pose release.
                self.assertEqual(
                    [result.gesture for result in results],
                    [GestureState.IDLE] * 3 + [expected, GestureState.IDLE],
                )
                self.assertTrue(engine.diagnostics.two_finger_hand)

    def test_three_finger_swipes_change_windows(self) -> None:
        for start_x, end_x, expected in (
            (200.0, 0.0, GestureState.PREVIOUS_WINDOW),
            (0.0, 200.0, GestureState.NEXT_WINDOW),
        ):
            with self.subTest(start_x=start_x, end_x=end_x):
                engine = GestureEngine()
                engine.evaluate(make_three_finger_hand(300.0, wrist_x=start_x))

                results = []
                step = (end_x - start_x) / 5
                for frame in range(1, 6):
                    results.append(
                        engine.evaluate(
                            make_three_finger_hand(
                                300.0,
                                wrist_x=start_x + step * frame,
                            )
                        )
                    )

                self.assertEqual(
                    [result.gesture for result in results],
                    [GestureState.IDLE] * 3 + [expected, GestureState.IDLE],
                )
                diagnostics = engine.diagnostics
                self.assertTrue(diagnostics.three_finger_hand)
                self.assertFalse(diagnostics.two_finger_hand)
                self.assertEqual(diagnostics.stable_pose, HandPose.THREE_FINGER)

    def test_two_finger_pose_is_not_three_finger_pose(self) -> None:
        engine = GestureEngine()

        engine.evaluate(make_index_middle_hand(300.0))

        self.assertTrue(engine.diagnostics.two_finger_hand)
        self.assertFalse(engine.diagnostics.three_finger_hand)

    def test_two_finger_return_stroke_is_ignored_but_repeat_works(
        self,
    ) -> None:
        engine = GestureEngine()

        engine.evaluate(make_index_middle_hand(300.0, wrist_x=0.0))
        self.assertEqual(
            engine.evaluate(make_index_middle_hand(300.0, wrist_x=200.0)).gesture,
            GestureState.NEXT_TAB,
        )
        # Bringing the hand back does not switch straight back again.
        for wrist_x in (100.0, 0.0, -100.0):
            self.assertEqual(
                engine.evaluate(
                    make_index_middle_hand(300.0, wrist_x=wrist_x)
                ).gesture,
                GestureState.IDLE,
            )
        # Swiping the same way again switches again, without releasing.
        self.assertEqual(
            engine.evaluate(make_index_middle_hand(300.0, wrist_x=100.0)).gesture,
            GestureState.NEXT_TAB,
        )

    def test_two_finger_vertical_movement_does_not_scroll(self) -> None:
        current_time = [0.0]
        engine = GestureEngine(clock=lambda: current_time[0])

        engine.evaluate(make_index_middle_hand(300.0))
        for wrist_y in (280.0, 260.0, 240.0, 220.0):
            current_time[0] += 0.05
            result = engine.evaluate(make_index_middle_hand(wrist_y))
            self.assertEqual(result.gesture, GestureState.IDLE)
            self.assertEqual(result.scroll_amount, 0)

    def test_right_pinch_is_not_mistaken_for_two_finger_pose(self) -> None:
        hand = make_index_middle_hand(300.0)
        points = list(hand.keypoints)
        middle_tip = points[12]
        points[4] = Keypoint(
            4, middle_tip.x, middle_tip.y, 1.0, 0.5, 0.5
        )
        engine = GestureEngine()

        engine.evaluate(make_open_hand(300.0))
        result = confirm_pinch(
            engine, HandPrediction(tuple(points), hand.confidence)
        )

        self.assertFalse(engine.diagnostics.two_finger_hand)
        self.assertEqual(result.gesture, GestureState.PINCH_RIGHT_CLICK)

    def test_scroll_and_two_finger_tab_swipe_dispatch_through_translator(
        self,
    ) -> None:
        current_time = [0.0]
        engine = GestureEngine(clock=lambda: current_time[0])
        hotkeys = []
        scrolls = []
        translator = CommandTranslator(
            clock=lambda: current_time[0],
            emit=lambda _message: None,
            hotkey=lambda *keys: hotkeys.append(keys),
            scroll=scrolls.append,
        )

        def feed(hand):
            current_time[0] += 0.05
            result = engine.evaluate(hand)
            translator.process(result.gesture, result.scroll_amount)

        # Open-palm scrolling with sideways drift never switches tabs.
        feed(make_open_hand(300.0))
        for wrist_y in (280.0, 260.0, 240.0, 230.0):
            feed(make_open_hand(wrist_y))
        for wrist_x in (140.0, 200.0, 260.0, 320.0):
            feed(make_open_hand(230.0, wrist_x=wrist_x))
        self.assertTrue(scrolls)
        self.assertEqual(hotkeys, [])

        # A two-finger swipe switches tabs and never scrolls.
        scroll_count = len(scrolls)
        current_time[0] += 1.0
        feed(make_index_middle_hand(300.0, wrist_x=320.0))
        for wrist_x in (260.0, 200.0, 140.0, 80.0, 20.0):
            feed(make_index_middle_hand(300.0, wrist_x=wrist_x))
        self.assertEqual(hotkeys, [("ctrl", "shift", "tab")])
        self.assertEqual(len(scrolls), scroll_count)

    def test_scroll_with_sideways_drift_never_switches_tabs(self) -> None:
        current_time = [0.0]
        engine = GestureEngine(clock=lambda: current_time[0])
        engine.evaluate(make_open_hand(300.0, wrist_x=100.0))

        gestures = []
        # Lock to scrolling, then drift far enough sideways for a tab swipe.
        for wrist_x, wrist_y in (
            (100.0, 280.0),
            (100.0, 260.0),
            (140.0, 250.0),
            (200.0, 245.0),
            (260.0, 245.0),
        ):
            current_time[0] += 0.05
            gestures.append(
                engine.evaluate(
                    make_open_hand(wrist_y, wrist_x=wrist_x)
                ).gesture
            )

        self.assertIn(GestureState.SCROLL_DOWN, gestures)
        self.assertNotIn(GestureState.NEXT_TAB, gestures)
        self.assertNotIn(GestureState.PREVIOUS_TAB, gestures)

    def test_scroll_disarms_pinches_until_open_hand_is_presented_again(
        self,
    ) -> None:
        for accidental_pinch, expected_gesture in (
            (
                make_hand(100, 120, middle_x=130),
                GestureState.PINCH_LEFT_CLICK,
            ),
            (
                make_hand(100, 200, middle_x=130),
                GestureState.PINCH_RIGHT_CLICK,
            ),
        ):
            with self.subTest(gesture=expected_gesture):
                current_time = [0.0]
                engine = GestureEngine(clock=lambda: current_time[0])

                engine.evaluate(make_open_hand(300.0))
                self.assertTrue(engine.diagnostics.pinch_enabled)

                current_time[0] = 0.05
                initial_movement = engine.evaluate(make_open_hand(290.0))
                self.assertEqual(initial_movement.gesture, GestureState.IDLE)
                self.assertTrue(engine.diagnostics.pinch_enabled)
                current_time[0] = 0.1
                scroll_result = engine.evaluate(make_open_hand(280.0))
                self.assertEqual(
                    scroll_result.gesture,
                    GestureState.SCROLL_DOWN,
                )
                self.assertFalse(engine.diagnostics.pinch_enabled)

                self.assertEqual(
                    engine.evaluate(accidental_pinch).gesture,
                    GestureState.IDLE,
                )
                self.assertEqual(
                    engine.evaluate(accidental_pinch).gesture,
                    GestureState.IDLE,
                )

                engine.evaluate(make_open_hand(290.0))
                self.assertTrue(engine.diagnostics.pinch_enabled)
                self.assertEqual(
                    confirm_pinch(engine, accidental_pinch).gesture,
                    expected_gesture,
                )

    def test_index_only_swipe_right_to_left_goes_back(self) -> None:
        engine = GestureEngine()

        engine.evaluate(make_index_only_hand(300.0, wrist_x=200.0))
        result = engine.evaluate(
            make_index_only_hand(300.0, wrist_x=0.0)
        )

        self.assertEqual(result.gesture, GestureState.BACK)

    def test_index_only_swipe_left_to_right_goes_forward(self) -> None:
        engine = GestureEngine()

        engine.evaluate(make_index_only_hand(300.0, wrist_x=0.0))
        result = engine.evaluate(
            make_index_only_hand(300.0, wrist_x=200.0)
        )

        self.assertEqual(result.gesture, GestureState.FORWARD)

    def test_index_and_middle_swipe_switches_tabs_not_history(self) -> None:
        engine = GestureEngine()

        engine.evaluate(make_index_middle_hand(300.0, wrist_x=200.0))
        result = engine.evaluate(
            make_index_middle_hand(300.0, wrist_x=0.0)
        )

        self.assertEqual(result.gesture, GestureState.PREVIOUS_TAB)
        self.assertFalse(engine.diagnostics.index_only_hand)

    def test_index_only_swipe_does_not_require_thumb_visibility(self) -> None:
        engine = GestureEngine()
        start = make_index_only_hand(300.0, wrist_x=200.0)
        start_points = list(start.keypoints)
        thumb = start_points[4]
        start_points[4] = Keypoint(
            4, thumb.x, thumb.y, 0.1, thumb.normalized_x, thumb.normalized_y
        )
        end = make_index_only_hand(300.0, wrist_x=0.0)
        end_points = list(end.keypoints)
        thumb = end_points[4]
        end_points[4] = Keypoint(
            4, thumb.x, thumb.y, 0.1, thumb.normalized_x, thumb.normalized_y
        )

        engine.evaluate(HandPrediction(tuple(start_points), start.confidence))
        result = engine.evaluate(HandPrediction(tuple(end_points), end.confidence))

        self.assertEqual(result.gesture, GestureState.BACK)

    def test_index_only_pose_suppresses_right_pinch_until_swipe(self) -> None:
        engine = GestureEngine()
        start = make_index_only_hand_with_right_pinch(
            300.0,
            wrist_x=200.0,
        )
        end = make_index_only_hand_with_right_pinch(
            300.0,
            wrist_x=0.0,
        )

        self.assertEqual(engine.evaluate(start).gesture, GestureState.IDLE)
        self.assertEqual(engine.evaluate(end).gesture, GestureState.BACK)

    def test_bent_non_index_fingers_are_down_even_when_tips_are_far_from_mcp(
        self,
    ) -> None:
        engine = GestureEngine()

        engine.evaluate(make_bent_index_only_hand(300.0, wrist_x=200.0))
        result = engine.evaluate(
            make_bent_index_only_hand(300.0, wrist_x=0.0)
        )

        self.assertEqual(result.gesture, GestureState.BACK)

    def test_vertical_open_hand_movement_scrolls_without_switching_tabs(
        self,
    ) -> None:
        current_time = [0.0]
        engine = GestureEngine(clock=lambda: current_time[0])

        engine.evaluate(make_open_hand(300.0))
        current_time[0] = 0.05
        result = engine.evaluate(make_open_hand(220.0))

        self.assertEqual(result.gesture, GestureState.SCROLL_DOWN)
        self.assertLess(result.scroll_amount, 0)

    def test_diagonal_open_hand_movement_neither_scrolls_nor_changes_tabs(
        self,
    ) -> None:
        current_time = [0.0]
        engine = GestureEngine(clock=lambda: current_time[0])

        engine.evaluate(make_open_hand(300.0, wrist_x=0.0))
        current_time[0] = 0.05
        result = engine.evaluate(
            make_open_hand(140.0, wrist_x=200.0)
        )

        self.assertEqual(result.gesture, GestureState.IDLE)
        self.assertEqual(result.scroll_amount, 0)

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

        engine = GestureEngine()
        engine.evaluate(make_open_hand(300.0))
        result = confirm_pinch(
            engine, HandPrediction(tuple(points), hand.confidence)
        )

        self.assertEqual(result.gesture, GestureState.PINCH_LEFT_CLICK)

    def test_pointer_pose_is_distinct_from_other_poses(self) -> None:
        engine = GestureEngine()
        for hand, expected in (
            (make_pointer_hand(), True),
            (make_open_hand(300.0), False),
            (make_index_only_hand(300.0), False),
            (make_index_middle_hand(300.0), False),
        ):
            engine.evaluate(hand)
            self.assertEqual(engine.diagnostics.pointer_pose, expected)

    def test_pointer_starts_only_after_pose_is_held(self) -> None:
        engine = GestureEngine()

        for frame in range(1, POINTER_ENTER_FRAMES):
            result = engine.evaluate(make_pointer_hand())
            self.assertIsNone(result.pointer)
            self.assertEqual(
                engine.diagnostics.pointer_arming_frames,
                frame,
            )

        result = engine.evaluate(make_pointer_hand())
        self.assertTrue(engine.diagnostics.pointer_active)
        self.assertIsNotNone(result.pointer)
        self.assertFalse(result.pointer.frozen)
        self.assertEqual(result.gesture, GestureState.IDLE)

    def test_pointer_sample_follows_palm_centre(self) -> None:
        engine = GestureEngine()
        start_pointer(engine)

        first = engine.evaluate(make_pointer_hand(wrist_x=100.0)).pointer
        moved = engine.evaluate(make_pointer_hand(wrist_x=160.0)).pointer

        self.assertAlmostEqual(moved.x - first.x, 60.0)
        self.assertAlmostEqual(moved.hand_size, 100.0)

    def test_pointer_pinch_clicks_without_open_palm_arming(self) -> None:
        engine = GestureEngine()
        start_pointer(engine)
        self.assertFalse(engine.diagnostics.pinch_enabled)

        claw = make_pointer_hand()
        index_tip = claw.keypoints[8]
        left = confirm_pinch(
            engine, with_thumb_at(claw, index_tip.x, index_tip.y)
        )
        self.assertEqual(left.gesture, GestureState.PINCH_LEFT_CLICK)
        self.assertTrue(left.pointer.frozen)

        self.assertEqual(
            engine.evaluate(make_pointer_hand()).gesture,
            GestureState.IDLE,
        )
        middle_tip = claw.keypoints[12]
        right = confirm_pinch(
            engine, with_thumb_at(claw, middle_tip.x + 5.0, middle_tip.y)
        )
        self.assertEqual(right.gesture, GestureState.PINCH_RIGHT_CLICK)

    def test_pointer_freezes_while_pinch_is_forming(self) -> None:
        engine = GestureEngine()
        start_pointer(engine)
        claw = make_pointer_hand()
        index_tip = claw.keypoints[8]

        # Thumb 0.6 hand sizes from the index tip: not a click yet, frozen.
        result = engine.evaluate(
            with_thumb_at(claw, index_tip.x - 60.0, index_tip.y)
        )

        self.assertEqual(result.gesture, GestureState.IDLE)
        self.assertTrue(result.pointer.frozen)
        self.assertTrue(engine.diagnostics.pointer_frozen)
        self.assertFalse(engine.evaluate(claw).pointer.frozen)

    def test_pointer_session_ignores_swipe_poses_and_scrolling(self) -> None:
        engine = GestureEngine()
        start_pointer(engine, wrist_x=200.0)

        for wrist_x in (150.0, 100.0, 50.0, 0.0):
            result = engine.evaluate(
                make_index_middle_hand(300.0, wrist_x=wrist_x)
            )
            self.assertEqual(result.gesture, GestureState.IDLE)
            self.assertIsNotNone(result.pointer)
        for wrist_x in (50.0, 100.0, 150.0, 200.0):
            result = engine.evaluate(
                make_index_only_hand(300.0, wrist_x=wrist_x)
            )
            self.assertEqual(result.gesture, GestureState.IDLE)

    def test_pointer_ends_when_ring_and_pinky_open(self) -> None:
        current_time = [0.0]
        engine = GestureEngine(clock=lambda: current_time[0])
        start_pointer(engine)

        # Brief loss is tolerated: the session survives a short gap.
        for _ in range(POINTER_EXIT_FRAMES - 1):
            result = engine.evaluate(None)
            self.assertIsNone(result.pointer)
            self.assertTrue(engine.diagnostics.pointer_active)
        engine.evaluate(make_pointer_hand())
        self.assertTrue(engine.diagnostics.pointer_active)

        for _ in range(POINTER_EXIT_FRAMES):
            engine.evaluate(make_open_hand(300.0))
        self.assertFalse(engine.diagnostics.pointer_active)

        # Back in normal mode, the open palm scrolls again.
        engine.evaluate(make_open_hand(300.0))
        current_time[0] += 0.05
        engine.evaluate(make_open_hand(290.0))
        current_time[0] += 0.05
        self.assertEqual(
            engine.evaluate(make_open_hand(280.0)).gesture,
            GestureState.SCROLL_DOWN,
        )

    def test_brief_pointer_like_transition_does_not_start_pointer(self) -> None:
        engine = GestureEngine()

        for _ in range(POINTER_ENTER_FRAMES - 1):
            engine.evaluate(make_pointer_hand(wrist_x=200.0))
        engine.evaluate(make_index_middle_hand(300.0, wrist_x=200.0))

        self.assertFalse(engine.diagnostics.pointer_active)
        self.assertEqual(
            engine.evaluate(make_index_middle_hand(300.0, wrist_x=0.0)).gesture,
            GestureState.PREVIOUS_TAB,
        )

    def test_low_confidence_fist_landmark_does_not_block_pinch(self) -> None:
        hand = make_hand(100, 120)
        points = list(hand.keypoints)
        points[6] = Keypoint(6, 100.0, 100.0, 0.1, 0.5, 0.5)

        engine = GestureEngine()
        engine.evaluate(make_open_hand(300.0))
        result = confirm_pinch(
            engine, HandPrediction(tuple(points), hand.confidence)
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


class GestureConfusionTests(unittest.TestCase):
    """Transitions and noise that must not be taken for another gesture."""

    def test_detection_glitch_does_not_reset_a_swipe_in_progress(self) -> None:
        engine = GestureEngine()

        engine.evaluate(make_index_only_hand(300.0, wrist_x=200.0))
        engine.evaluate(make_index_only_hand(300.0, wrist_x=150.0))
        engine.evaluate(None)
        engine.evaluate(make_hand(50.0, 150.0))
        self.assertEqual(engine.diagnostics.stable_pose, HandPose.INDEX_ONLY)

        result = engine.evaluate(make_index_only_hand(300.0, wrist_x=50.0))
        self.assertEqual(result.gesture, GestureState.BACK)

    def test_dropout_does_not_stop_scrolling(self) -> None:
        current_time = [0.0]
        engine = GestureEngine(clock=lambda: current_time[0])

        for wrist_y in (300.0, 290.0, 280.0):
            result = engine.evaluate(make_open_hand(wrist_y))
            current_time[0] += 0.05
        self.assertEqual(result.gesture, GestureState.SCROLL_DOWN)
        engine.evaluate(None)
        current_time[0] += 0.05

        result = engine.evaluate(make_open_hand(270.0))
        self.assertEqual(result.gesture, GestureState.SCROLL_DOWN)

    def test_stray_frame_of_another_pose_does_not_take_over(self) -> None:
        current_time = [0.0]
        engine = GestureEngine(clock=lambda: current_time[0])

        gestures = []
        for hand in (
            make_index_middle_hand(300.0, wrist_x=200.0),
            make_index_middle_hand(300.0, wrist_x=160.0),
            make_open_hand(300.0, wrist_x=120.0),
            make_index_middle_hand(300.0, wrist_x=80.0),
            make_index_middle_hand(300.0, wrist_x=40.0),
        ):
            current_time[0] += 0.05
            result = engine.evaluate(hand)
            gestures.append(result.gesture)
            self.assertEqual(result.scroll_amount, 0)
            self.assertEqual(
                engine.diagnostics.stable_pose, HandPose.TWO_FINGER
            )

        self.assertEqual(
            gestures, [GestureState.IDLE] * 4 + [GestureState.PREVIOUS_TAB]
        )

    def test_single_noisy_pinch_frame_does_not_click(self) -> None:
        engine = GestureEngine()
        engine.evaluate(make_open_hand(300.0))

        self.assertEqual(
            engine.evaluate(make_hand(100, 120)).gesture, GestureState.IDLE
        )
        self.assertEqual(
            engine.evaluate(make_hand(50, 150)).gesture, GestureState.IDLE
        )
        # The noise did not use up the open-palm arming.
        self.assertTrue(engine.diagnostics.pinch_enabled)
        self.assertEqual(
            confirm_pinch(engine, make_hand(100, 120)).gesture,
            GestureState.PINCH_LEFT_CLICK,
        )

    def _pointer_left_feed(self, engine: GestureEngine):
        claw = make_pointer_hand()
        index_tip = claw.keypoints[8]

        def feed(left_ratio: float) -> GestureState:
            hand = with_thumb_at(
                claw, index_tip.x - left_ratio * 100.0, index_tip.y
            )
            return engine.evaluate(hand).gesture

        return feed

    def test_second_pinch_in_double_click_window_is_looser(self) -> None:
        current_time = [0.0]
        engine = GestureEngine(clock=lambda: current_time[0])
        start_pointer(engine)
        feed = self._pointer_left_feed(engine)

        # A normal first click.
        feed(0.2)
        self.assertEqual(feed(0.2), GestureState.PINCH_LEFT_CLICK)
        current_time[0] = 0.3
        feed(1.0)
        self.assertTrue(engine.diagnostics.double_click_ready)
        # Within the window, 0.5 (above PINCH_THRESHOLD) clicks on its
        # first frame.
        current_time[0] = 0.6
        self.assertEqual(feed(0.5), GestureState.PINCH_LEFT_CLICK)
        # The second pinch closes the window.
        self.assertFalse(engine.diagnostics.double_click_ready)

    def test_loose_pinch_outside_double_click_window_does_not_click(
        self,
    ) -> None:
        current_time = [0.0]
        engine = GestureEngine(clock=lambda: current_time[0])
        start_pointer(engine)
        feed = self._pointer_left_feed(engine)

        feed(0.2)
        feed(0.2)
        feed(1.0)
        current_time[0] = 2.0  # window over
        for _ in range(3):
            self.assertEqual(feed(0.5), GestureState.IDLE)
        self.assertFalse(engine.diagnostics.double_click_ready)

    def test_loose_pinch_without_a_first_click_does_not_click(self) -> None:
        engine = GestureEngine()
        start_pointer(engine)
        feed = self._pointer_left_feed(engine)

        for _ in range(3):
            self.assertEqual(feed(0.5), GestureState.IDLE)

    def test_held_pinch_flicker_clicks_once(self) -> None:
        clicks = []
        translator = CommandTranslator(
            cooldown_sec=0.0,
            emit=lambda _message: None,
            click=clicks.append,
            double_click=lambda button: clicks.append(f"double-{button}"),
        )
        engine = GestureEngine()
        start_pointer(engine)
        claw = make_pointer_hand()
        index_tip = claw.keypoints[8]

        def feed(left_ratio: float) -> GestureState:
            hand = with_thumb_at(
                claw, index_tip.x - left_ratio * 100.0, index_tip.y
            )
            result = engine.evaluate(hand)
            translator.process(result.gesture)
            return result.gesture

        # Ratios between PINCH_THRESHOLD and PINCH_RELEASE_THRESHOLD keep
        # the pinch held instead of releasing and clicking again.
        for ratio in (0.2, 0.2, 0.5, 0.2, 0.55, 0.2):
            feed(ratio)
        self.assertEqual(clicks, ["left"])

        # Released and pinched again right away: a double-click.
        self.assertEqual(feed(1.0), GestureState.IDLE)
        feed(0.2)
        feed(0.2)
        self.assertEqual(clicks, ["left", "double-left"])

    def test_held_left_pinch_does_not_turn_into_right_click(self) -> None:
        engine = GestureEngine()
        start_pointer(engine)
        claw = make_pointer_hand()
        index_tip = claw.keypoints[8]
        middle_tip = claw.keypoints[12]

        confirm_pinch(
            engine, with_thumb_at(claw, index_tip.x - 20.0, index_tip.y)
        )
        # The thumb slides toward the middle tip while still near the index.
        for _ in range(3):
            result = engine.evaluate(
                with_thumb_at(claw, middle_tip.x + 5.0, middle_tip.y)
            )
            self.assertEqual(result.gesture, GestureState.PINCH_LEFT_CLICK)

    def test_moving_claw_does_not_start_pointer(self) -> None:
        engine = GestureEngine()

        # 0.4 hand sizes per frame: a swipe made with bent fingers.
        for frame in range(POINTER_ENTER_FRAMES * 2):
            engine.evaluate(make_pointer_hand(wrist_x=400.0 - 40.0 * frame))
            self.assertFalse(engine.diagnostics.pointer_active)

        for _ in range(POINTER_ENTER_FRAMES):
            engine.evaluate(make_pointer_hand(wrist_x=0.0))
        self.assertTrue(engine.diagnostics.pointer_active)

    def test_pointer_waits_until_a_swipe_pose_has_ended(self) -> None:
        engine = GestureEngine()
        engine.evaluate(make_index_only_hand(300.0))

        for _ in range(POSE_RELEASE_FRAMES - 1):
            engine.evaluate(make_pointer_hand())
            self.assertEqual(engine.diagnostics.pointer_arming_frames, 0)
        for _ in range(POINTER_ENTER_FRAMES - 1):
            engine.evaluate(make_pointer_hand())
            self.assertFalse(engine.diagnostics.pointer_active)
        engine.evaluate(make_pointer_hand())
        self.assertTrue(engine.diagnostics.pointer_active)

    def test_slow_drift_does_not_swipe(self) -> None:
        current_time = [0.0]
        engine = GestureEngine(clock=lambda: current_time[0])

        # One hand size per second: too slow to be a swipe.
        for step in range(31):
            result = engine.evaluate(
                make_index_middle_hand(300.0, wrist_x=10.0 * step)
            )
            self.assertEqual(result.gesture, GestureState.IDLE)
            current_time[0] += 0.1

        result = engine.evaluate(make_index_middle_hand(300.0, wrist_x=450.0))
        self.assertEqual(result.gesture, GestureState.NEXT_TAB)

    def test_diagonal_swipe_does_not_change_tabs(self) -> None:
        engine = GestureEngine()

        engine.evaluate(make_index_middle_hand(300.0, wrist_x=0.0))
        result = engine.evaluate(make_index_middle_hand(130.0, wrist_x=150.0))

        self.assertEqual(result.gesture, GestureState.IDLE)

    def test_history_return_stroke_is_ignored_until_lockout_ends(self) -> None:
        current_time = [0.0]
        engine = GestureEngine(clock=lambda: current_time[0])

        def swipe_to(wrist_x: float, at: float) -> GestureState:
            current_time[0] = at
            return engine.evaluate(
                make_index_only_hand(300.0, wrist_x=wrist_x)
            ).gesture

        swipe_to(200.0, 0.0)
        self.assertEqual(swipe_to(0.0, 0.1), GestureState.BACK)
        self.assertEqual(swipe_to(200.0, 0.3), GestureState.IDLE)

        after_lockout = 0.1 + SWIPE_RETURN_LOCKOUT_SEC + 0.5
        swipe_to(0.0, after_lockout)
        self.assertEqual(
            swipe_to(200.0, after_lockout + 0.1), GestureState.FORWARD
        )

    def test_pinch_arming_expires(self) -> None:
        current_time = [0.0]
        engine = GestureEngine(clock=lambda: current_time[0])
        engine.evaluate(make_open_hand(300.0))
        self.assertTrue(engine.diagnostics.pinch_enabled)

        current_time[0] = PINCH_ARM_TIMEOUT_SEC + 0.1
        result = confirm_pinch(engine, make_hand(100, 120))

        self.assertEqual(result.gesture, GestureState.IDLE)
        self.assertFalse(engine.diagnostics.pinch_enabled)

    def test_swipe_pose_clears_pinch_arming(self) -> None:
        engine = GestureEngine()
        engine.evaluate(make_open_hand(300.0))
        for _ in range(2):
            engine.evaluate(make_index_only_hand(300.0))
        self.assertEqual(engine.diagnostics.stable_pose, HandPose.INDEX_ONLY)
        self.assertFalse(engine.diagnostics.pinch_enabled)

        for _ in range(POSE_RELEASE_FRAMES + PINCH_CONFIRM_FRAMES):
            self.assertEqual(
                engine.evaluate(make_hand(100, 120)).gesture,
                GestureState.IDLE,
            )

def with_depth(
    hand: HandPrediction,
    toward_camera: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> HandPrediction:
    """Add MediaPipe-style depth; curl each finger toward the camera.

    The DIP joint sits midway between PIP and tip in the image, so the 2D
    shape is unchanged. A finger with depth ``d`` has its DIP ``d`` and its
    tip ``3d`` closer to the camera, a total flexion of ``atan(2d / 20)``
    for the test hands' 40-pixel PIP-to-tip span (``d = 10`` gives 45°).
    """
    points = [replace(point, z=0.0) for point in hand.keypoints]
    for finger, (pip_index, dip_index, tip_index) in enumerate(
        ((6, 7, 8), (10, 11, 12), (14, 15, 16), (18, 19, 20))
    ):
        pip = points[pip_index]
        tip = points[tip_index]
        depth = toward_camera[finger]
        points[dip_index] = Keypoint(
            dip_index,
            (pip.x + tip.x) / 2,
            (pip.y + tip.y) / 2,
            1.0,
            0.5,
            0.5,
            z=-depth,
        )
        points[tip_index] = replace(tip, z=-3 * depth)
    return HandPrediction(tuple(points), hand.confidence)


def make_depth_claw(wrist_x: float = 100.0) -> HandPrediction:
    """A claw that looks exactly like the two-finger pose in 2D."""
    return with_depth(
        make_index_middle_hand(300.0, wrist_x=wrist_x),
        toward_camera=(10.0, 10.0, 0.0, 0.0),
    )


class DepthClawTests(unittest.TestCase):
    """With depth, the claw is told apart by how far fingers bend in 3D."""

    def test_claw_bent_toward_camera_is_the_pointer_pose(self) -> None:
        engine = GestureEngine()

        engine.evaluate(make_depth_claw())

        diagnostics = engine.diagnostics
        self.assertAlmostEqual(diagnostics.finger_flex[0], 45.0)
        self.assertTrue(diagnostics.pointer_pose)
        self.assertFalse(diagnostics.two_finger_hand)
        self.assertFalse(diagnostics.index_only_hand)

    def test_depth_claw_starts_and_moves_the_cursor(self) -> None:
        engine = GestureEngine()

        for _ in range(POINTER_ENTER_FRAMES):
            result = engine.evaluate(make_depth_claw())
        self.assertTrue(engine.diagnostics.pointer_active)

        moved = engine.evaluate(make_depth_claw(wrist_x=140.0)).pointer
        self.assertAlmostEqual(moved.x - result.pointer.x, 40.0)

    def test_straight_fingers_with_depth_keep_their_poses(self) -> None:
        engine = GestureEngine()
        engine.evaluate(with_depth(make_index_middle_hand(300.0)))
        self.assertTrue(engine.diagnostics.two_finger_hand)
        self.assertFalse(engine.diagnostics.pointer_pose)

        engine = GestureEngine()
        engine.evaluate(with_depth(make_index_middle_hand(300.0, 200.0)))
        result = engine.evaluate(
            with_depth(make_index_middle_hand(300.0, wrist_x=0.0))
        )
        self.assertEqual(result.gesture, GestureState.PREVIOUS_TAB)

        engine = GestureEngine()
        engine.evaluate(with_depth(make_index_only_hand(300.0, 200.0)))
        result = engine.evaluate(
            with_depth(make_index_only_hand(300.0, wrist_x=0.0))
        )
        self.assertEqual(result.gesture, GestureState.BACK)

    def test_slightly_bent_fingers_are_not_a_claw(self) -> None:
        engine = GestureEngine()

        # 5 px toward the camera: about 27°, below the claw threshold.
        engine.evaluate(
            with_depth(
                make_index_middle_hand(300.0),
                toward_camera=(5.0, 5.0, 0.0, 0.0),
            )
        )

        self.assertLess(
            engine.diagnostics.finger_flex[0], POINTER_CLAW_MIN_FLEX_DEGREES
        )
        self.assertFalse(engine.diagnostics.pointer_pose)
        self.assertTrue(engine.diagnostics.two_finger_hand)

    def test_claw_pinch_clicks(self) -> None:
        engine = GestureEngine()
        for _ in range(POINTER_ENTER_FRAMES):
            engine.evaluate(make_depth_claw())
        claw = make_depth_claw()
        index_tip = claw.keypoints[8]

        result = confirm_pinch(
            engine, with_thumb_at(claw, index_tip.x - 20.0, index_tip.y)
        )

        self.assertEqual(result.gesture, GestureState.PINCH_LEFT_CLICK)
        self.assertTrue(result.pointer.frozen)

    def test_one_noisy_depth_frame_keeps_the_claw(self) -> None:
        engine = GestureEngine()
        engine.evaluate(make_depth_claw())

        # Depth noise makes the fingers read only 27° bent for one frame;
        # the smoothed value stays inside the claw band.
        engine.evaluate(
            with_depth(
                make_index_middle_hand(300.0),
                toward_camera=(5.0, 5.0, 0.0, 0.0),
            )
        )

        self.assertTrue(engine.diagnostics.pointer_pose)
        self.assertFalse(engine.diagnostics.two_finger_hand)

    def test_brief_dropout_pauses_pointer_arming(self) -> None:
        engine = GestureEngine()
        for _ in range(POINTER_ENTER_FRAMES - 2):
            engine.evaluate(make_depth_claw())
        engine.evaluate(None)
        self.assertEqual(
            engine.diagnostics.pointer_arming_frames, POINTER_ENTER_FRAMES - 2
        )

        engine.evaluate(make_depth_claw())
        self.assertFalse(engine.diagnostics.pointer_active)
        engine.evaluate(make_depth_claw())
        self.assertTrue(engine.diagnostics.pointer_active)

if __name__ == "__main__":
    unittest.main()
