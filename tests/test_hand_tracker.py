import unittest

from src.recognition.hand_detector import HandPrediction, Keypoint
from src.recognition.hand_tracker import HandTracker, palm_center


def hand_at(x: float, y: float) -> HandPrediction:
    """A 21-point hand whose landmarks all sit at one normalized point."""
    return HandPrediction(
        tuple(
            Keypoint(index, x * 640, y * 480, 1.0, x, y) for index in range(21)
        ),
        1.0,
    )


class HandTrackerTests(unittest.TestCase):
    def test_palm_center_is_normalized(self) -> None:
        self.assertEqual(palm_center(hand_at(0.3, 0.6)), (0.3, 0.6))

    def test_locks_onto_the_first_hand(self) -> None:
        tracker = HandTracker()
        first = hand_at(0.3, 0.5)

        tracked, ignored = tracker.update([first])

        self.assertIs(tracked, first)
        self.assertEqual(ignored, [])
        self.assertTrue(tracker.locked)

    def test_keeps_the_first_hand_when_detector_order_swaps(self) -> None:
        tracker = HandTracker()
        tracker.update([hand_at(0.3, 0.5)])

        for x in (0.32, 0.34, 0.36):
            mine, other = hand_at(x, 0.5), hand_at(0.8, 0.5)
            tracked, ignored = tracker.update([other, mine])
            self.assertIs(tracked, mine)
            self.assertEqual(ignored, [other])

    def test_short_dropout_does_not_switch_hands(self) -> None:
        tracker = HandTracker(lost_frames=3)
        tracker.update([hand_at(0.3, 0.5)])
        other = hand_at(0.8, 0.5)

        for _ in range(3):
            tracked, ignored = tracker.update([other])
            self.assertIsNone(tracked)
            self.assertEqual(ignored, [other])

        mine = hand_at(0.31, 0.5)
        tracked, _ = tracker.update([other, mine])
        self.assertIs(tracked, mine)

    def test_other_hand_takes_over_once_the_first_is_gone(self) -> None:
        tracker = HandTracker(lost_frames=2)
        tracker.update([hand_at(0.3, 0.5)])
        other = hand_at(0.8, 0.5)

        for _ in range(3):
            tracker.update([other])
        tracked, ignored = tracker.update([other])

        self.assertIs(tracked, other)
        self.assertEqual(ignored, [])

    def test_lock_is_released_when_no_hand_is_seen(self) -> None:
        tracker = HandTracker(lost_frames=1)
        tracker.update([hand_at(0.3, 0.5)])

        tracker.update([])
        tracker.update([])

        self.assertFalse(tracker.locked)


if __name__ == "__main__":
    unittest.main()
