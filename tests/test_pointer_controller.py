import random
import sys
import unittest
from unittest.mock import Mock, patch

from src.config import (
    POINTER_FAST_PX_PER_HAND_SIZE,
    POINTER_SLOW_PX_PER_HAND_SIZE,
)
from src.recognition.gesture_engine import PointerSample
from src.translation.pointer_controller import (
    OneEuroFilter,
    PointerController,
    _move_cursor_by,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def sample(x: float, y: float = 200.0, frozen: bool = False) -> PointerSample:
    return PointerSample(x, y, 100.0, frozen)


class PointerControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.moves: list[tuple[int, int]] = []
        self.controller = PointerController(
            move_by=lambda dx, dy: self.moves.append((dx, dy)),
            clock=self.clock,
        )

    def feed(self, *samples: PointerSample | None, dt: float = 1 / 30) -> None:
        for item in samples:
            self.clock.now += dt
            self.controller.update(item)

    def total(self) -> tuple[int, int]:
        return (
            sum(dx for dx, _ in self.moves),
            sum(dy for _, dy in self.moves),
        )

    def test_first_sample_only_anchors(self) -> None:
        self.feed(sample(300.0))
        self.assertEqual(self.moves, [])

    def test_hand_moving_right_in_image_moves_cursor_left(self) -> None:
        # The camera is not mirrored: image-right is the user's left.
        self.feed(*(sample(300.0 + 5.0 * step) for step in range(20)))
        dx, dy = self.total()
        self.assertLess(dx, 0)
        self.assertEqual(dy, 0)

    def test_vertical_motion_is_not_inverted(self) -> None:
        self.feed(*(sample(300.0, 200.0 + 5.0 * step) for step in range(20)))
        self.assertGreater(self.total()[1], 0)

    def test_frozen_samples_do_not_move_and_unfreezing_does_not_jump(
        self,
    ) -> None:
        self.feed(sample(300.0))
        self.feed(*(sample(300.0 - 4.0 * step, frozen=True) for step in range(10)))
        self.assertEqual(self.moves, [])

        self.feed(sample(260.0), sample(260.0))
        dx, _ = self.total()
        self.assertLessEqual(abs(dx), 2)

    def test_losing_tracking_repositions_without_moving(self) -> None:
        self.feed(sample(300.0), None, sample(100.0), sample(100.0))
        dx, _ = self.total()
        self.assertLessEqual(abs(dx), 1)

    def test_fast_motion_moves_further_per_hand_size(self) -> None:
        controller = self.controller
        self.assertEqual(controller.gain(0.0), POINTER_SLOW_PX_PER_HAND_SIZE)
        self.assertEqual(controller.gain(100.0), POINTER_FAST_PX_PER_HAND_SIZE)

        slow_moves = []
        slow = PointerController(
            move_by=lambda dx, dy: slow_moves.append(dx),
            clock=self.clock,
        )
        for step in range(61):  # one hand size over two seconds
            self.clock.now += 1 / 30
            slow.update(sample(300.0 - 100.0 * step / 60))
        for _ in range(30):
            self.clock.now += 1 / 30
            slow.update(sample(200.0))

        fast_moves = []
        fast = PointerController(
            move_by=lambda dx, dy: fast_moves.append(dx),
            clock=self.clock,
        )
        for step in range(7):  # one hand size in 0.2 seconds
            self.clock.now += 1 / 30
            fast.update(sample(300.0 - 100.0 * step / 6))
        for _ in range(30):
            self.clock.now += 1 / 30
            fast.update(sample(200.0))

        self.assertGreater(sum(fast_moves), sum(slow_moves) * 1.5)

    def test_resting_jitter_barely_moves_cursor(self) -> None:
        rng = random.Random(1)
        self.feed(
            *(
                sample(300.0 + rng.uniform(-1.5, 1.5), 200.0 + rng.uniform(-1.5, 1.5))
                for _ in range(90)
            )
        )
        dx, dy = self.total()
        travelled = sum(abs(mx) + abs(my) for mx, my in self.moves)
        self.assertLess(travelled, 40)
        self.assertLess(abs(dx) + abs(dy), 15)


class OneEuroFilterTests(unittest.TestCase):
    def test_follows_a_step_and_converges(self) -> None:
        smoother = OneEuroFilter()
        smoother(0.0, 0.0)
        value = 0.0
        for _ in range(60):
            value = smoother(100.0, 1 / 30)
        self.assertAlmostEqual(value, 100.0, delta=1.0)


class MoveCursorTests(unittest.TestCase):
    def test_moves_relative_and_stays_off_screen_corners(self) -> None:
        fake_pyautogui = Mock()
        fake_pyautogui.size.return_value = (1920, 1080)
        fake_pyautogui.position.return_value = (5, 5)

        with patch.dict(sys.modules, {"pyautogui": fake_pyautogui}):
            _move_cursor_by(-50, 10)

        fake_pyautogui.moveTo.assert_called_once_with(1, 15)
        self.assertEqual(fake_pyautogui.PAUSE, 0.0)


if __name__ == "__main__":
    unittest.main()
