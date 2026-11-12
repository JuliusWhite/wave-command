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
from ..recognition.gesture_engine import (
    FIST_CONFIRM_FRAMES,
    FIST_REARM_FRAMES,
    FIST_RELEASE_FRAMES,
    GestureDiagnostics,
    GestureState,
)
from ..recognition.hand_detector import HandPrediction


@dataclass(frozen=True)
class FrameState:
    hands: list[HandPrediction]
    gesture: GestureState
    pinch_distance_ratio: float | None
    alert: str | None
    diagnostics: GestureDiagnostics | None = None


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
    _POINTER_COLOR = (255, 200, 80)
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
            pinch_landmarks = None
            is_pointing = (
                hand_index == 0
                and state.gesture == GestureState.POINTER_MOVE
            )
            if hand_index == 0:
                if state.gesture == GestureState.PINCH_LEFT_CLICK:
                    pinch_landmarks = (4, 8)
                elif state.gesture == GestureState.PINCH_RIGHT_CLICK:
                    pinch_landmarks = (8, 12)
            self._draw_hand(
                frame,
                hand,
                pinch_landmarks=pinch_landmarks,
                is_pointing=is_pointing,
            )

        if state.gesture in {
            GestureState.PINCH_LEFT_CLICK,
            GestureState.PINCH_RIGHT_CLICK,
        }:
            gesture_color = self._PINCH_COLOR
        elif state.gesture == GestureState.POINTER_MOVE:
            gesture_color = self._POINTER_COLOR
        else:
            gesture_color = self._TEXT_COLOR
        distance_text = (
            f"{state.pinch_distance_ratio:.2f}"
            if state.pinch_distance_ratio is not None
            else "--"
        )
        diagnostics = state.diagnostics or GestureDiagnostics()
        curl_text = "/".join(
            "--" if ratio is None else f"{ratio:.2f}"
            for ratio in diagnostics.curl_ratios
        )
        left_pinch_text = (
            "--"
            if diagnostics.left_pinch_ratio is None
            else f"{diagnostics.left_pinch_ratio:.2f}"
        )
        right_pinch_text = (
            "--"
            if diagnostics.right_pinch_ratio is None
            else f"{diagnostics.right_pinch_ratio:.2f}"
        )
        thumb_curl_text = (
            "--"
            if diagnostics.thumb_curl_ratio is None
            else f"{diagnostics.thumb_curl_ratio:.2f}"
        )
        if diagnostics.fist_active:
            filter_text = (
                "Filter: fist active, release "
                f"{diagnostics.fist_release_frames}/{FIST_RELEASE_FRAMES}"
            )
        elif diagnostics.fist_rearm_pending:
            filter_text = (
                "Filter: neutral rearm "
                f"{diagnostics.fist_rearm_frames}/{FIST_REARM_FRAMES}"
            )
        elif diagnostics.fist_confirmation_frames:
            filter_text = (
                "Filter: fist confirm "
                f"{diagnostics.fist_confirmation_frames}/{FIST_CONFIRM_FRAMES}"
            )
        else:
            filter_text = "Filter: ready"
        hud_lines = (
            (f"Wave Command | {self.backend_name}", self._TEXT_COLOR),
            (f"FPS: {fps:.1f}", self._TEXT_COLOR),
            (f"Gesture: {state.gesture.value}", gesture_color),
            (f"Raw: {diagnostics.raw_gesture.value}", self._TEXT_COLOR),
            (filter_text, self._TEXT_COLOR),
            (
                "Curl I/M/R/P: "
                f"{curl_text} "
                f"({diagnostics.curled_fingers}/{diagnostics.visible_fingers})",
                self._TEXT_COLOR,
            ),
            (f"Thumb curl: {thumb_curl_text}", self._TEXT_COLOR),
            (
                f"Pinch T-I/I-M: {left_pinch_text}/{right_pinch_text} "
                f"(threshold {PINCH_THRESHOLD:.2f})",
                self._TEXT_COLOR,
            ),
            (f"Selected pinch ratio: {distance_text}", self._TEXT_COLOR),
            (
                "Point: move | Pinch: click | Fist: close window | Q: quit",
                self._TEXT_COLOR,
            ),
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
        pinch_landmarks: tuple[int, int] | None,
        is_pointing: bool,
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

        line_start_index, line_end_index = pinch_landmarks or (4, 8)
        line_start = points[line_start_index]
        line_end = points[line_end_index]
        if is_pointing:
            color = self._POINTER_COLOR
        elif pinch_landmarks is not None:
            color = self._PINCH_COLOR
        else:
            color = self._OPEN_COLOR
        cv2.line(
            frame,
            (round(line_start.x), round(line_start.y)),
            (round(line_end.x), round(line_end.y)),
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
