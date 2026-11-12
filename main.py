"""Wave Command entry point: webcam -> recognition -> mouse actions."""

from __future__ import annotations

import time

from src.config import HUD_ALERT_DURATION_SEC
from src.recognition import GestureEngine, HandDetector
from src.translation import CommandTranslator
from src.webcam import FrameState, Visualizer


def main() -> None:
    detector = HandDetector()
    gesture_engine = GestureEngine()
    translator = CommandTranslator()
    last_alert: str | None = None
    last_alert_at = 0.0
    visualizer = Visualizer(backend_name=detector.backend_name)

    def process_frame(frame: object) -> FrameState:
        nonlocal last_alert, last_alert_at

        hands = detector.detect(frame)
        visualizer.backend_name = detector.backend_name
        gesture_result = gesture_engine.evaluate(hands[0] if hands else None)
        alert = translator.process(gesture_result.gesture)
        if alert:
            last_alert = alert
            last_alert_at = time.monotonic()

        visible_alert = (
            last_alert
            if time.monotonic() - last_alert_at < HUD_ALERT_DURATION_SEC
            else None
        )
        return FrameState(
            hands=hands,
            gesture=gesture_result.gesture,
            pinch_distance_ratio=gesture_result.pinch_distance_ratio,
            alert=visible_alert,
            diagnostics=gesture_engine.diagnostics,
        )

    try:
        visualizer.run(process_frame)
    finally:
        detector.close()


if __name__ == "__main__":
    main()
