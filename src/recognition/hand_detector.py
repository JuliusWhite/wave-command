"""Convert webcam frames into a consistent set of 21 hand landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import (
    CONFIDENCE_THRESHOLD,
    YOLO_HAND_MODEL_PATH,
)


@dataclass(frozen=True)
class Keypoint:
    """One hand landmark, with pixel and normalized image coordinates."""

    id: int
    x: float
    y: float
    conf: float
    normalized_x: float
    normalized_y: float


@dataclass(frozen=True)
class HandPrediction:
    """The 21 landmarks for one detected hand."""

    keypoints: tuple[Keypoint, ...]
    confidence: float


class HandDetector:
    """Use optional custom YOLO hand-pose weights or MediaPipe Hands."""

    def __init__(
        self,
        model_path: Path = YOLO_HAND_MODEL_PATH,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self._backend: str
        self._model: Any = None

        if self.model_path.is_file():
            try:
                from ultralytics import YOLO
            except ImportError:
                print(
                    "[DETECTOR] Ultralytics is unavailable; using MediaPipe "
                    "instead of the optional YOLO model."
                )
            else:
                self._model = YOLO(str(self.model_path))
                self._backend = "YOLO"
                return

        self._init_mediapipe()

    @property
    def backend_name(self) -> str:
        """Return the active landmark-detection backend."""
        return self._backend

    def _init_mediapipe(self) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is required when no usable custom YOLO hand model "
                "is available. Install requirements.txt and run "
                "scripts/setup_models.py."
            ) from exc

        self._model = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=self.confidence_threshold,
            min_tracking_confidence=self.confidence_threshold,
        )
        self._backend = "MediaPipe"

    def detect(self, frame: Any) -> list[HandPrediction]:
        """Detect hands in a BGR image and return landmarks in image pixels."""
        if self._backend == "YOLO":
            return self._detect_yolo(frame)
        return self._detect_mediapipe(frame)

    def _detect_mediapipe(self, frame: Any) -> list[HandPrediction]:
        import cv2

        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._model.process(rgb_frame)
        predictions: list[HandPrediction] = []

        if not result.multi_hand_landmarks:
            return predictions

        handedness = result.multi_handedness or []
        for hand_index, hand_landmarks in enumerate(result.multi_hand_landmarks):
            confidence = self._mediapipe_hand_confidence(handedness, hand_index)
            if confidence < self.confidence_threshold:
                continue

            keypoints = tuple(
                Keypoint(
                    id=index,
                    x=landmark.x * width,
                    y=landmark.y * height,
                    conf=confidence,
                    normalized_x=landmark.x,
                    normalized_y=landmark.y,
                )
                for index, landmark in enumerate(hand_landmarks.landmark)
            )
            if len(keypoints) == 21:
                predictions.append(HandPrediction(keypoints, confidence))

        return predictions

    @staticmethod
    def _mediapipe_hand_confidence(handedness: list[Any], hand_index: int) -> float:
        if hand_index >= len(handedness):
            return 1.0
        classifications = handedness[hand_index].classification
        if not classifications:
            return 1.0
        return float(classifications[0].score)

    def _detect_yolo(self, frame: Any) -> list[HandPrediction]:
        height, width = frame.shape[:2]
        results = self._model.predict(
            source=frame,
            conf=self.confidence_threshold,
            verbose=False,
        )
        predictions: list[HandPrediction] = []

        if any(
            result.keypoints is None
            or result.keypoints.xy.ndim != 3
            or result.keypoints.xy.shape[1] != 21
            for result in results
        ):
            print(
                "[DETECTOR] The YOLO weights do not expose 21 pose keypoints; "
                "switching to MediaPipe."
            )
            self._init_mediapipe()
            return self._detect_mediapipe(frame)

        for result in results:
            if result.keypoints is None:
                continue

            points_by_hand = result.keypoints.xy.cpu().tolist()
            point_confidences = (
                result.keypoints.conf.cpu().tolist()
                if result.keypoints.conf is not None
                else None
            )
            box_confidences = (
                result.boxes.conf.cpu().tolist()
                if result.boxes is not None
                else []
            )

            for hand_index, points in enumerate(points_by_hand):
                if len(points) != 21:
                    continue
                hand_confidence = (
                    float(box_confidences[hand_index])
                    if hand_index < len(box_confidences)
                    else 1.0
                )
                if hand_confidence < self.confidence_threshold:
                    continue

                confidence_values = (
                    point_confidences[hand_index]
                    if point_confidences is not None
                    else [hand_confidence] * 21
                )
                keypoints = tuple(
                    Keypoint(
                        id=index,
                        x=float(point[0]),
                        y=float(point[1]),
                        conf=float(confidence_values[index]),
                        normalized_x=float(point[0]) / width,
                        normalized_y=float(point[1]) / height,
                    )
                    for index, point in enumerate(points)
                )
                predictions.append(HandPrediction(keypoints, hand_confidence))

        return predictions

    def close(self) -> None:
        """Release backend resources when supported."""
        if self._backend == "MediaPipe" and self._model is not None:
            self._model.close()
