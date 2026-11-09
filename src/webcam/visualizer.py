"""OpenCV camera loop and visualization of recognized hand gestures."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..config import (
    CAMERA_ID,
    HAND_CONNECTIONS,
    PINCH_THRESHOLD,
    WINDOW_NAME,
)
from ..recognition.gesture_engine import GestureState
from ..recognition.hand_detector import HandPrediction


@dataclass(frozen=True)
class FrameState:
    hands: list[HandPrediction]
    gesture: GestureState
    pinch_distance_ratio: float | None
    alert: str | None


class Visualizer:
    """Own webcam capture and draw landmarks, gesture state, and HUD."""

    _FINGER_COLORS = (
        (255, 160, 70),   # thumb, BGR
        (80, 220, 255),   # index
        (255, 210, 80),   # middle
        (210, 100, 255),  # ring
        (100, 230, 120),  # pinky
    )
    _PINCH_COLOR = (80, 230, 80)
    _OPEN_COLOR = (40, 100, 255)
    _TEXT_COLOR = (245, 245, 245)

    def __init__(self, camera_id: int = CAMERA_ID, backend_name: str = "") -> None:
        self.camera_id = camera_id
        self.backend_name = backend_name

    def run(self, process_frame: Callable[[Any], FrameState]) -> None:
        """Capture frames and pass each frame to the application pipeline."""
        import cv2

        camera = cv2.VideoCapture(self.camera_id)
        if not camera.isOpened():
            camera.release()
            raise RuntimeError(
                f"Could not open camera {self.camera_id}. Check the camera "
                "connection or update CAMERA_ID in src/config.py."
            )

        previous_frame_at = time.perf_counter()
        try:
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("The webcam stopped returning frames.")

                state = process_frame(frame)
                now = time.perf_counter()
                elapsed = now - previous_frame_at
                fps = 1.0 / elapsed if elapsed > 0 else 0.0
                previous_frame_at = now

                self._draw(frame, state, fps)
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()

    def _draw(self, frame: Any, state: FrameState, fps: float) -> None:
        import cv2

        for hand_index, hand in enumerate(state.hands):
            self._draw_hand(
                frame,
                hand,
                is_pinching=(
                    hand_index == 0
                    and state.gesture == GestureState.PINCH_CLOSE_WINDOW
                ),
            )

        gesture_color = (
            self._PINCH_COLOR
            if state.gesture == GestureState.PINCH_CLOSE_WINDOW
            else self._TEXT_COLOR
        )
        distance_text = (
            f"{state.pinch_distance_ratio:.2f}"
            if state.pinch_distance_ratio is not None
            else "--"
        )
        hud_lines = (
            (f"Wave Command | {self.backend_name}", self._TEXT_COLOR),
            (f"FPS: {fps:.1f}", self._TEXT_COLOR),
            (f"Gesture: {state.gesture.value}", gesture_color),
            (f"Pinch ratio: {distance_text} (threshold {PINCH_THRESHOLD:.2f})",
             self._TEXT_COLOR),
            ("Dry-run only | Press Q to quit", self._TEXT_COLOR),
        )
        for line_index, (text, color) in enumerate(hud_lines):
            cv2.putText(
                frame,
                text,
                (12, 28 + line_index * 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
                cv2.LINE_AA,
            )

        if state.alert:
            cv2.rectangle(frame, (8, frame.shape[0] - 58),
                          (frame.shape[1] - 8, frame.shape[0] - 8), (30, 90, 30), -1)
            cv2.putText(
                frame,
                state.alert,
                (18, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                self._TEXT_COLOR,
                2,
                cv2.LINE_AA,
            )

    def _draw_hand(
        self,
        frame: Any,
        hand: HandPrediction,
        is_pinching: bool,
    ) -> None:
        import cv2

        points = hand.keypoints
        for start, end in HAND_CONNECTIONS:
            color = self._finger_color(start, end)
            cv2.line(
                frame,
                (round(points[start].x), round(points[start].y)),
                (round(points[end].x), round(points[end].y)),
                color,
                2,
                cv2.LINE_AA,
            )

        for point in points:
            cv2.circle(
                frame,
                (round(point.x), round(point.y)),
                4,
                self._finger_color(point.id, point.id),
                -1,
                cv2.LINE_AA,
            )

        thumb_tip = points[4]
        index_tip = points[8]
        color = self._PINCH_COLOR if is_pinching else self._OPEN_COLOR
        cv2.line(
            frame,
            (round(thumb_tip.x), round(thumb_tip.y)),
            (round(index_tip.x), round(index_tip.y)),
            color,
            3,
            cv2.LINE_AA,
        )

    def _finger_color(self, start: int, end: int) -> tuple[int, int, int]:
        landmark_id = max(start, end)
        if landmark_id <= 4:
            finger_index = 0
        elif landmark_id <= 8:
            finger_index = 1
        elif landmark_id <= 12:
            finger_index = 2
        elif landmark_id <= 16:
            finger_index = 3
        else:
            finger_index = 4
        return self._FINGER_COLORS[finger_index]
