import unittest

from src.recognition.gesture_engine import GestureEngine, GestureState
from src.recognition.hand_detector import HandPrediction, Keypoint


def make_hand(thumb_x: float, index_x: float) -> HandPrediction:
    points = [
        Keypoint(i, 100.0, 100.0, 1.0, 0.5, 0.5)
        for i in range(21)
    ]
    points[0] = Keypoint(0, 100.0, 100.0, 1.0, 0.5, 0.5)
    points[4] = Keypoint(4, thumb_x, 100.0, 1.0, 0.5, 0.5)
    points[8] = Keypoint(8, index_x, 100.0, 1.0, 0.5, 0.5)
    points[9] = Keypoint(9, 100.0, 200.0, 1.0, 0.5, 0.5)
    return HandPrediction(tuple(points), 1.0)


class GestureEngineTests(unittest.TestCase):
    def test_detects_close_thumb_and_index_tips(self) -> None:
        result = GestureEngine(pinch_threshold=0.35).evaluate(make_hand(100, 120))
        self.assertEqual(result.gesture, GestureState.PINCH_CLOSE_WINDOW)
        self.assertAlmostEqual(result.pinch_distance_ratio, 0.2)

    def test_open_distance_is_idle(self) -> None:
        result = GestureEngine(pinch_threshold=0.35).evaluate(make_hand(50, 150))
        self.assertEqual(result.gesture, GestureState.IDLE)

    def test_missing_hand_is_idle(self) -> None:
        result = GestureEngine().evaluate(None)
        self.assertEqual(result.gesture, GestureState.IDLE)
        self.assertIsNone(result.pinch_distance_ratio)


if __name__ == "__main__":
    unittest.main()
