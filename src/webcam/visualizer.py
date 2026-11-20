"""OpenCV camera loop and visualization of recognized hand gestures."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..config import (
    CAMERA_ID,
    HAND_CONNECTIONS,
    PINCH_THRESHOLD,
    POINTER_CLAW_MAX_FLEX_DEGREES,
    POINTER_CLAW_MIN_FLEX_DEGREES,
    POINTER_ENTER_FRAMES,
    WINDOW_NAME,
)
from ..recognition.gesture_engine import (
    FIST_CONFIRM_FRAMES,
    FIST_REARM_FRAMES,
    FIST_RELEASE_FRAMES,
    GestureDiagnostics,
    GestureState,
    HandPose,
)
from ..recognition.hand_detector import HandPrediction
from .overlay import OverlayStatus

Color = tuple[int, int, int]
FOOTER_TEXT = "B: bypass | D: details | Q: quit"
OVERLAY_FOOTER_TEXT = "B: bypass | C: hide camera | D: details | Q: quit"


@dataclass(frozen=True)
class FrameState:
    hands: list[HandPrediction]
    gesture: GestureState
    pinch_distance_ratio: float | None
    alert: str | None
    diagnostics: GestureDiagnostics | None = None
    # Hands in view that are not the tracked one; drawn greyed out.
    ignored_hands: list[HandPrediction] = field(default_factory=list)


class Visualizer:
    """Own webcam capture and draw landmarks, gesture state, and HUD.

    By default the HUD is compact: one large mode label with a short hint,
    the last action, and a key reminder. Press D to toggle the detailed
    tuning readout.

    With a status ``overlay`` (see ``StatusOverlay``) the same status is
    shown in an always-on-top window and the camera preview becomes
    optional: C or closing the preview hides it, and the overlay's
    "camera" button brings it back. Without an overlay the preview is
    always shown.

    ``bypassed`` turns gesture actions off while detection and the HUD keep
    running; B or the overlay's "bypass" button toggles it. The application
    pipeline reads it to skip cursor movement and commands.
    """

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
    _MUTED_COLOR = (170, 170, 170)
    _POINTER_COLOR = (255, 200, 60)
    _SCROLL_COLOR = (80, 200, 255)
    _NAVIGATION_COLOR = (230, 140, 255)
    _TABS_COLOR = (250, 139, 167)  # #a78bfa lavender violet
    _OUTLINE_COLOR = (0, 0, 0)
    _PANEL_COLOR = (20, 20, 20)
    _PANEL_OPACITY = 0.6
    _ALERT_COLOR = (40, 110, 40)
    # The overlay shows neutral modes (READY, NO HAND) in the brand blues
    # instead of the HUD's white and grey; mode colors are kept as is.
    _OVERLAY_NEUTRAL_COLORS = {
        _TEXT_COLOR: (255, 208, 92),  # #5cd0ff electric blue
        _MUTED_COLOR: (204, 167, 126),  # #7ea7cc soft blue-grey
    }
    _FONT_WIDTH_PER_SCALE = 19  # approx. pixels per character at scale 1.0

    def __init__(
        self,
        camera_id: int = CAMERA_ID,
        backend_name: str = "",
        show_details: bool = False,
        overlay: Any = None,
        show_camera: bool = True,
        bypassed: bool = False,
    ) -> None:
        self.camera_id = camera_id
        self.backend_name = backend_name
        self.show_details = show_details
        self.overlay = overlay
        # Without an overlay the preview is the only feedback and way to quit.
        self.show_camera = show_camera or overlay is None
        self.bypassed = bypassed

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
        preview_open = False
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

                if self.overlay is not None:
                    if self.overlay.consume_camera_toggle():
                        self.show_camera = not self.show_camera
                    if self.overlay.consume_bypass_toggle():
                        self.bypassed = not self.bypassed
                    self.overlay.show(self.overlay_status(state))
                    self.overlay.pump()
                    if self.overlay.quit_requested:
                        break

                if not self.show_camera:
                    if preview_open:
                        cv2.destroyWindow(WINDOW_NAME)
                        preview_open = False
                    continue

                self._draw(frame, state, fps)
                cv2.imshow(WINDOW_NAME, frame)
                preview_open = True
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("d"):
                    self.show_details = not self.show_details
                if key == ord("b"):
                    self.bypassed = not self.bypassed
                if self.overlay is not None and (
                    key == ord("c")
                    or cv2.getWindowProperty(
                        WINDOW_NAME, cv2.WND_PROP_VISIBLE
                    ) < 1
                ):
                    # Hide the preview; the overlay keeps reporting status.
                    self.show_camera = False
        finally:
            camera.release()
            cv2.destroyAllWindows()
            if self.overlay is not None:
                self.overlay.close()

    def overlay_status(self, state: FrameState) -> OverlayStatus:
        """Status for the always-on-top overlay, in the HUD's colors."""
        label, hint, color = self._status(
            state, state.diagnostics or GestureDiagnostics()
        )
        if state.hands:
            hand_text, hand_color = "Hand detected", self._PINCH_COLOR
        else:
            hand_text, hand_color = "No hand in view", self._OPEN_COLOR
        return OverlayStatus(
            label=label,
            hint=hint,
            color=self._OVERLAY_NEUTRAL_COLORS.get(color, color),
            hand_text=hand_text,
            hand_color=hand_color,
            action=self._short_alert(state.alert),
            camera_visible=self.show_camera,
            bypassed=self.bypassed,
        )

    def _draw(self, frame: Any, state: FrameState, fps: float) -> None:
        import cv2

        for hand in state.ignored_hands:
            self._draw_ignored_hand(cv2, frame, hand)
        for hand_index, hand in enumerate(state.hands):
            pinch_tip_index = None
            if hand_index == 0:
                if state.gesture == GestureState.PINCH_LEFT_CLICK:
                    pinch_tip_index = 8
                elif state.gesture == GestureState.PINCH_RIGHT_CLICK:
                    pinch_tip_index = 12
            self._draw_hand(
                frame,
                hand,
                pinch_tip_index=pinch_tip_index,
            )

        diagnostics = state.diagnostics or GestureDiagnostics()
        height, width = frame.shape[:2]
        if self.show_details:
            self._draw_details(cv2, frame, state, diagnostics, fps)
        else:
            self._draw_status(cv2, frame, state, diagnostics, fps, width)

        alert = self._short_alert(state.alert)
        if alert:
            text_width = self._text_width(alert, 0.8)
            left = max(8, (width - text_width) // 2 - 16)
            right = min(width - 8, left + text_width + 32)
            cv2.rectangle(
                frame,
                (left, height - 78),
                (right, height - 34),
                self._ALERT_COLOR,
                -1,
            )
            self._text(cv2, frame, alert, (left + 16, height - 46), 0.8)
        footer = FOOTER_TEXT if self.overlay is None else OVERLAY_FOOTER_TEXT
        self._text(
            cv2, frame, footer, (12, height - 12), 0.5, self._MUTED_COLOR
        )

    def _draw_status(
        self,
        cv2: Any,
        frame: Any,
        state: FrameState,
        diagnostics: GestureDiagnostics,
        fps: float,
        width: int,
    ) -> None:
        """Compact HUD: the current mode in large type and one hint."""
        label, hint, color = self._status(state, diagnostics)
        panel_width = max(
            self._text_width(label, 1.1), self._text_width(hint, 0.6)
        ) + 28
        self._shade(cv2, frame, (8, 8), (8 + panel_width, 92))
        self._text(cv2, frame, label, (20, 50), 1.1, color, thickness=3)
        self._text(cv2, frame, hint, (20, 80), 0.6)

        fps_text = f"{fps:.0f} FPS"
        self._text(
            cv2,
            frame,
            fps_text,
            (width - self._text_width(fps_text, 0.5) - 12, 24),
            0.5,
            self._MUTED_COLOR,
        )

    def _status(
        self,
        state: FrameState,
        diagnostics: GestureDiagnostics,
    ) -> tuple[str, str, Color]:
        """Return the mode label, a short hint, and the label color."""
        if self.bypassed:
            hand = "hand seen" if state.hands else "no hand"
            return (
                "BYPASSED",
                f"gesture actions off ({hand}) - B or resume",
                self._OPEN_COLOR,
            )
        if state.gesture == GestureState.PINCH_LEFT_CLICK:
            return "LEFT CLICK", "open fingers to release", self._PINCH_COLOR
        if state.gesture == GestureState.PINCH_RIGHT_CLICK:
            return "RIGHT CLICK", "open fingers to release", self._PINCH_COLOR
        if diagnostics.pointer_active:
            hint = (
                "frozen - finish the pinch to click"
                if diagnostics.pointer_frozen
                else "pinch thumb + index/middle to click"
            )
            return "CURSOR", hint, self._POINTER_COLOR
        if diagnostics.pointer_arming_frames:
            return (
                "CURSOR",
                "hold still... "
                f"{diagnostics.pointer_arming_frames}/{POINTER_ENTER_FRAMES}",
                self._MUTED_COLOR,
            )
        if diagnostics.stable_pose == HandPose.OPEN_PALM:
            hint = "move up / down"
            if diagnostics.pinch_enabled:
                hint += " | pinch to click"
            return "SCROLL", hint, self._SCROLL_COLOR
        if diagnostics.stable_pose == HandPose.INDEX_ONLY:
            return (
                "HISTORY",
                "swipe sideways: back / forward",
                self._NAVIGATION_COLOR,
            )
        if diagnostics.stable_pose == HandPose.TWO_FINGER:
            return (
                "TABS",
                "swipe sideways: previous / next",
                self._TABS_COLOR,
            )
        if not state.hands:
            return "NO HAND", "show your hand to the camera", self._MUTED_COLOR
        if diagnostics.pinch_enabled:
            return "READY", "pinch armed: pinch to click", self._TEXT_COLOR
        return (
            "READY",
            "claw: cursor | palm: scroll | 1-2 fingers: swipe",
            self._TEXT_COLOR,
        )

    @staticmethod
    def _short_alert(alert: str | None) -> str | None:
        """Turn a translator message into a short label like 'NEXT TAB'."""
        if not alert:
            return None
        match = re.search(r"Action: (\w+)", alert)
        if match is None:
            return alert
        return match.group(1).replace("_", " ")

    def _text_width(self, text: str, scale: float) -> int:
        # Estimated so the HUD needs no cv2.getTextSize round trip.
        return int(len(text) * self._FONT_WIDTH_PER_SCALE * scale)

    def _text(
        self,
        cv2: Any,
        frame: Any,
        text: str,
        origin: tuple[int, int],
        scale: float,
        color: Color | None = None,
        thickness: int = 1,
    ) -> None:
        """Draw text with a dark outline so it reads on any background."""
        for line_color, line_thickness in (
            (self._OUTLINE_COLOR, thickness + 3),
            (color or self._TEXT_COLOR, thickness),
        ):
            cv2.putText(
                frame,
                text,
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                line_color,
                line_thickness,
                cv2.LINE_AA,
            )

    def _shade(
        self,
        cv2: Any,
        frame: Any,
        top_left: tuple[int, int],
        bottom_right: tuple[int, int],
    ) -> None:
        """Darken a translucent panel behind text."""
        overlay = frame.copy()
        cv2.rectangle(overlay, top_left, bottom_right, self._PANEL_COLOR, -1)
        cv2.addWeighted(
            overlay,
            self._PANEL_OPACITY,
            frame,
            1.0 - self._PANEL_OPACITY,
            0,
            frame,
        )

    def _draw_details(
        self,
        cv2: Any,
        frame: Any,
        state: FrameState,
        diagnostics: GestureDiagnostics,
        fps: float,
    ) -> None:
        """Detailed tuning readout (toggle with D)."""
        label, hint, _ = self._status(state, diagnostics)
        if state.gesture in {
            GestureState.PINCH_LEFT_CLICK,
            GestureState.PINCH_RIGHT_CLICK,
        }:
            gesture_color = self._PINCH_COLOR
        else:
            gesture_color = self._TEXT_COLOR
        distance_text = (
            f"{state.pinch_distance_ratio:.2f}"
            if state.pinch_distance_ratio is not None
            else "--"
        )
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
        if diagnostics.fist_active:
            filter_text = (
                "Filter: tips together, release "
                f"{diagnostics.fist_release_frames}/{FIST_RELEASE_FRAMES}"
            )
        elif diagnostics.fist_rearm_pending:
            filter_text = (
                "Filter: neutral rearm "
                f"{diagnostics.fist_rearm_frames}/{FIST_REARM_FRAMES}"
            )
        elif diagnostics.fist_confirmation_frames:
            filter_text = (
                "Filter: tips together confirm "
                f"{diagnostics.fist_confirmation_frames}/{FIST_CONFIRM_FRAMES}"
            )
        else:
            filter_text = "Filter: ready"
        hud_lines = (
            (
                f"Wave Command | {self.backend_name} | {fps:.0f} FPS",
                self._TEXT_COLOR,
            ),
            (f"Mode: {label} - {hint}", self._TEXT_COLOR),
            (f"Gesture: {state.gesture.value}", gesture_color),
            (f"Raw: {diagnostics.raw_gesture.value}", self._TEXT_COLOR),
            (filter_text, self._TEXT_COLOR),
            (
                f"Pointer: {self._pointer_text(diagnostics)}",
                self._POINTER_COLOR
                if diagnostics.pointer_active
                else self._TEXT_COLOR,
            ),
            (f"Pose: {diagnostics.stable_pose.value}", self._TEXT_COLOR),
            (f"Pinch gate: {self._pinch_gate_text(diagnostics)}", self._TEXT_COLOR),
            (
                "Curl I/M/R/P: "
                f"{curl_text} "
                f"({diagnostics.curled_fingers}/{diagnostics.visible_fingers})",
                self._TEXT_COLOR,
            ),
            (self._flex_text(diagnostics), self._TEXT_COLOR),
            (
                "Index-only history pose: "
                f"{'detected' if diagnostics.index_only_hand else 'not detected'}",
                self._TEXT_COLOR,
            ),
            (
                "Two-finger tab pose: "
                f"{'detected' if diagnostics.two_finger_hand else 'not detected'}",
                self._TEXT_COLOR,
            ),
            (
                f"Pinch L/R: {left_pinch_text}/{right_pinch_text} "
                f"(threshold {PINCH_THRESHOLD:.2f})",
                self._TEXT_COLOR,
            ),
            (f"Selected pinch ratio: {distance_text}", self._TEXT_COLOR),
        )
        scale = 0.5
        line_height = 20
        panel_width = max(
            self._text_width(text, scale) for text, _ in hud_lines
        ) + 24
        self._shade(
            cv2,
            frame,
            (8, 8),
            (8 + panel_width, 18 + line_height * len(hud_lines)),
        )
        for line_index, (text, color) in enumerate(hud_lines):
            self._text(
                cv2,
                frame,
                text,
                (16, 28 + line_index * line_height),
                scale,
                color,
            )

    @staticmethod
    def _flex_text(diagnostics: GestureDiagnostics) -> str:
        if all(flex is None for flex in diagnostics.finger_flex):
            return "Flex 3D I/M/R/P: no depth (2D claw rules)"
        values = "/".join(
            "--" if flex is None else f"{flex:.0f}"
            for flex in diagnostics.finger_flex
        )
        return (
            f"Flex 3D I/M/R/P: {values} "
            f"(claw {POINTER_CLAW_MIN_FLEX_DEGREES:.0f}-"
            f"{POINTER_CLAW_MAX_FLEX_DEGREES:.0f})"
        )

    @staticmethod
    def _pinch_gate_text(diagnostics: GestureDiagnostics) -> str:
        if diagnostics.pinch_held:
            return "held (open to release)"
        if diagnostics.pointer_active:
            return "direct (pointing)"
        if diagnostics.pinch_enabled:
            return "armed"
        return "show upright palm"

    @staticmethod
    def _pointer_text(diagnostics: GestureDiagnostics) -> str:
        if diagnostics.pointer_active:
            return (
                "frozen for click"
                if diagnostics.pointer_frozen
                else "moving cursor"
            )
        if diagnostics.pointer_arming_frames:
            return (
                "arming "
                f"{diagnostics.pointer_arming_frames}/{POINTER_ENTER_FRAMES}"
            )
        return "off (show claw pose)"

    def _draw_hand(
        self,
        frame: Any,
        hand: HandPrediction,
        pinch_tip_index: int | None,
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
        pinch_tip = points[
            pinch_tip_index if pinch_tip_index is not None else 8
        ]
        if pinch_tip_index is not None:
            color = self._PINCH_COLOR
        else:
            color = self._OPEN_COLOR
        cv2.line(
            frame,
            (round(thumb_tip.x), round(thumb_tip.y)),
            (round(pinch_tip.x), round(pinch_tip.y)),
            color,
            3,
            cv2.LINE_AA,
        )

    def _draw_ignored_hand(
        self, cv2: Any, frame: Any, hand: HandPrediction
    ) -> None:
        """Thin grey skeleton for a hand that is in view but not tracked."""
        points = hand.keypoints
        for start, end in HAND_CONNECTIONS:
            cv2.line(
                frame,
                (round(points[start].x), round(points[start].y)),
                (round(points[end].x), round(points[end].y)),
                self._MUTED_COLOR,
                1,
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
