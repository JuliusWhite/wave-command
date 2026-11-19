"""Always-on-top status window, so the camera preview is optional.

The overlay is a small borderless panel that stays above other windows and
shows the current mode, a short hint, whether a hand is seen, and the last
action, in the same colors as the camera HUD. It is driven from the camera
loop: call ``show`` with the latest status and ``pump`` once per frame.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from ..config import OVERLAY_MARGIN_PX, OVERLAY_OPACITY, WINDOW_NAME

Color = tuple[int, int, int]  # BGR, as in OpenCV


@dataclass(frozen=True)
class OverlayStatus:
    label: str
    hint: str
    color: Color
    hand_text: str
    hand_color: Color
    action: str | None
    camera_visible: bool
    bypassed: bool = False


def bgr_to_hex(color: Color) -> str:
    """Convert an OpenCV BGR color to a Tk ``#rrggbb`` string."""
    blue, green, red = color
    return f"#{red:02x}{green:02x}{blue:02x}"


class StatusOverlay:
    """Borderless, always-on-top Tk window showing the recognizer status.

    Drag it anywhere with the mouse. "bypass" turns gesture actions off
    (and "resume" back on), "camera" shows or hides the camera preview, and
    the cross quits Wave Command.
    """

    # Brand palette: light, electric blue on a deep navy panel.
    _BACKGROUND = "#0b1626"
    _BORDER = "#2fa8ff"
    _TEXT = "#e8f6ff"
    _MUTED = "#7ea7cc"
    _LINK = "#5cd0ff"
    _ACTION_BACKGROUND = "#1e8fff"
    _ACTION_TEXT = "#ffffff"
    # Bypass stays warm on purpose so "actions off" is never missed.
    _BYPASS_ACTIVE = "#ff8a3d"
    _MIN_CONTENT_WIDTH = 390

    def __init__(self) -> None:
        import tkinter as tk

        self._tk = tk
        self.quit_requested = False
        self._camera_toggle_requested = False
        self._bypass_toggle_requested = False
        self._last_status: OverlayStatus | None = None
        self._drag_offset = (0, 0)

        root = tk.Tk()
        root.title(WINDOW_NAME)
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        try:
            root.attributes("-alpha", OVERLAY_OPACITY)
        except tk.TclError:
            pass
        root.configure(
            bg=self._BACKGROUND,
            highlightthickness=1,
            highlightbackground=self._BORDER,
            highlightcolor=self._BORDER,
        )
        root.protocol("WM_DELETE_WINDOW", self._request_quit)
        self._root = root

        self._accent = tk.Frame(root, width=8, bg=self._BORDER)
        self._accent.grid(row=0, column=0, sticky="ns")
        content = tk.Frame(root, bg=self._BACKGROUND, padx=12, pady=8)
        content.grid(row=0, column=1, sticky="nsew")
        root.grid_columnconfigure(1, minsize=self._MIN_CONTENT_WIDTH)
        content.grid_columnconfigure(0, weight=1)

        self._label = self._make_label(
            content, ("Segoe UI", 18, "bold"), self._MUTED
        )
        self._label.grid(row=0, column=0, sticky="w", padx=(0, 12))
        buttons = tk.Frame(content, bg=self._BACKGROUND)
        buttons.grid(row=0, column=1, sticky="ne")
        self._bypass_button = self._make_label(
            buttons, ("Segoe UI", 9, "underline"), self._LINK
        )
        self._bypass_button.configure(text="bypass", cursor="hand2")
        self._bypass_button.pack(side="left", padx=(0, 10))
        self._bypass_button.bind("<Button-1>", self._request_bypass_toggle)
        self._camera_button = self._make_label(
            buttons, ("Segoe UI", 9, "underline"), self._LINK
        )
        self._camera_button.configure(cursor="hand2")
        self._camera_button.pack(side="left", padx=(0, 10))
        self._camera_button.bind("<Button-1>", self._request_camera_toggle)
        close_button = self._make_label(
            buttons, ("Segoe UI", 11, "bold"), self._LINK
        )
        close_button.configure(text="✕", cursor="hand2")
        close_button.pack(side="left")
        close_button.bind("<Button-1>", lambda _event: self._request_quit())

        self._hint = self._make_label(content, ("Segoe UI", 10), self._TEXT)
        self._hint.configure(wraplength=self._MIN_CONTENT_WIDTH, justify="left")
        self._hint.grid(row=1, column=0, columnspan=2, sticky="w")
        self._hand = self._make_label(
            content, ("Segoe UI", 10, "bold"), self._MUTED
        )
        self._hand.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self._action = self._make_label(
            content, ("Segoe UI", 11, "bold"), self._TEXT
        )
        self._action.configure(padx=8, pady=2)
        self._action.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

        for widget in (root, self._accent, content, self._label, self._hint,
                       self._hand, self._action, buttons):
            widget.bind("<ButtonPress-1>", self._start_drag, add="+")
            widget.bind("<B1-Motion>", self._drag, add="+")

        root.update_idletasks()
        width = root.winfo_reqwidth()
        x = root.winfo_screenwidth() - width - OVERLAY_MARGIN_PX
        root.geometry(f"+{max(0, x)}+{OVERLAY_MARGIN_PX}")
        root.update_idletasks()
        self._make_non_activating()

    def show(self, status: OverlayStatus) -> None:
        """Display a status; widgets are only touched when it changes."""
        if status == self._last_status:
            return
        self._last_status = status
        color = bgr_to_hex(status.color)
        self._accent.configure(bg=color)
        self._label.configure(text=status.label, fg=color)
        self._hint.configure(text=status.hint)
        self._hand.configure(
            text=f"●  {status.hand_text}", fg=bgr_to_hex(status.hand_color)
        )
        if status.action:
            self._action.configure(
                text=status.action,
                bg=self._ACTION_BACKGROUND,
                fg=self._ACTION_TEXT,
            )
        else:
            self._action.configure(
                text="no recent action", bg=self._BACKGROUND, fg=self._MUTED
            )
        self._camera_button.configure(
            text="hide camera" if status.camera_visible else "show camera"
        )
        if status.bypassed:
            self._bypass_button.configure(
                text="resume",
                fg=self._BYPASS_ACTIVE,
                font=("Segoe UI", 9, "bold underline"),
            )
        else:
            self._bypass_button.configure(
                text="bypass",
                fg=self._LINK,
                font=("Segoe UI", 9, "underline"),
            )

    def pump(self) -> None:
        """Process pending window events; call once per camera frame."""
        if self.quit_requested:
            return
        try:
            self._root.update()
        except self._tk.TclError:
            self.quit_requested = True

    def consume_camera_toggle(self) -> bool:
        """Return True once after the "camera" button was clicked."""
        requested = self._camera_toggle_requested
        self._camera_toggle_requested = False
        return requested

    def consume_bypass_toggle(self) -> bool:
        """Return True once after the "bypass"/"resume" button was clicked."""
        requested = self._bypass_toggle_requested
        self._bypass_toggle_requested = False
        return requested

    def close(self) -> None:
        try:
            self._root.destroy()
        except self._tk.TclError:
            pass

    def _make_label(self, parent: Any, font: tuple, color: str) -> Any:
        return self._tk.Label(
            parent, font=font, fg=color, bg=self._BACKGROUND, anchor="w"
        )

    def _request_quit(self) -> None:
        self.quit_requested = True

    def _request_camera_toggle(self, _event: Any) -> str:
        self._camera_toggle_requested = True
        return "break"  # a click on the button does not start a drag

    def _request_bypass_toggle(self, _event: Any) -> str:
        self._bypass_toggle_requested = True
        return "break"  # a click on the button does not start a drag

    def _start_drag(self, event: Any) -> None:
        self._drag_offset = (
            event.x_root - self._root.winfo_x(),
            event.y_root - self._root.winfo_y(),
        )

    def _drag(self, event: Any) -> None:
        dx, dy = self._drag_offset
        self._root.geometry(f"+{event.x_root - dx}+{event.y_root - dy}")

    def _make_non_activating(self) -> None:
        """On Windows, keep clicks on the overlay from stealing focus.

        Gesture shortcuts (tab and history keys) go to the focused window, so
        the overlay must never take focus from the browser. It is also hidden
        from the taskbar and Alt+Tab.
        """
        if sys.platform != "win32":
            return
        try:
            import ctypes

            user32 = ctypes.windll.user32
            gwl_exstyle = -20
            ws_ex_toolwindow = 0x00000080
            ws_ex_noactivate = 0x08000000
            hwnd = user32.GetParent(self._root.winfo_id()) or self._root.winfo_id()
            style = user32.GetWindowLongW(hwnd, gwl_exstyle)
            user32.SetWindowLongW(
                hwnd, gwl_exstyle, style | ws_ex_toolwindow | ws_ex_noactivate
            )
        except Exception:  # cosmetic; the overlay works without it
            pass
