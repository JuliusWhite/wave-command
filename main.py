"""Wave Command entry point: webcam -> recognition -> mouse and keys."""

from __future__ import annotations

import time

from src.config import (
    HUD_ALERT_DURATION_SEC,
    SHOW_CAMERA_PREVIEW,
    SHOW_STATUS_OVERLAY,
)
from src.recognition import GestureEngine, HandDetector, HandTracker
from src.translation import CommandTranslator, PointerController
from src.webcam import FrameState, StatusOverlay, Visualizer


def create_overlay() -> StatusOverlay | None:
    """Open the always-on-top status window, or None if Tk is unavailable."""
    if not SHOW_STATUS_OVERLAY:
        return None
    try:
        return StatusOverlay()
    except Exception as error:  # e.g. no Tk install or no display
        print(f"[OVERLAY] Status window unavailable ({error}); using the "
              "camera preview instead.")
        return None


def main() -> None:
    detector = HandDetector()
    tracker = HandTracker()
    gesture_engine = GestureEngine()
    translator = CommandTranslator()
    pointer = PointerController()
    last_alert: str | None = None
    last_alert_at = 0.0
    visualizer = Visualizer(
        backend_name=detector.backend_name,
        overlay=create_overlay(),
        show_camera=SHOW_CAMERA_PREVIEW,
    )

    def process_frame(frame: object) -> FrameState:
        nonlocal last_alert, last_alert_at

        tracked, ignored = tracker.update(detector.detect(frame))
        visualizer.backend_name = detector.backend_name
        gesture_result = gesture_engine.evaluate(tracked)
        if visualizer.bypassed:
            # Keep recognizing so the HUD stays live, but send nothing.
            pointer.update(None)
            translator.observe(gesture_result.gesture)
            alert = None
        else:
            pointer.update(gesture_result.pointer)
            alert = translator.process(
                gesture_result.gesture,
                scroll_amount=gesture_result.scroll_amount,
            )
        if alert:
            last_alert = alert
            last_alert_at = time.monotonic()

        visible_alert = (
            last_alert
            if time.monotonic() - last_alert_at < HUD_ALERT_DURATION_SEC
            else None
        )
        return FrameState(
            hands=[tracked] if tracked is not None else [],
            gesture=gesture_result.gesture,
            pinch_distance_ratio=gesture_result.pinch_distance_ratio,
            alert=visible_alert,
            diagnostics=gesture_engine.diagnostics,
            ignored_hands=ignored,
        )

    try:
        visualizer.run(process_frame)
    finally:
        detector.close()


if __name__ == "__main__":
    main()
