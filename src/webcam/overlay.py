"""Always-on-top status window, so the camera preview is optional.

The overlay is a small borderless panel that stays above other windows and
shows, as icons, the current mode, the last action, and whether a hand is
seen, in the same colors as the camera HUD. It is driven from the camera
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


# Icons come from Windows' Segoe Fluent Icons (Windows 11) or Segoe MDL2
# Assets (Windows 10); elsewhere plain Unicode symbols stand in. Each entry
# is (icon-font glyph, Unicode fallback, small badge text).
Icon = tuple[str, str, str]
_ICON_FONTS = ("Segoe Fluent Icons", "Segoe MDL2 Assets")
_MODE_ICONS: dict[str, Icon] = {
    "READY": ("", "✓", ""),          # check mark in a circle
    "NO HAND": ("", "⊘", ""),        # eye, crossed out
    "CURSOR": ("", "➚", ""),         # pointer arrow
    "LEFT CLICK": ("", "◧", "L"),    # mouse
    "RIGHT CLICK": ("", "◨", "R"),
    "SCROLL": ("", "⇕", ""),         # up/down chevrons
    "HISTORY": ("", "↶", ""),        # clock with arrow
    "TABS": ("", "⧉", ""),           # stacked pages
    "BYPASSED": ("", "⏸", ""),       # pause
}
_ACTION_ICONS: dict[str, Icon] = {
    "LEFT CLICK": ("", "◧", "L"),
    "RIGHT CLICK": ("", "◨", "R"),
    "NEXT TAB": ("", "⧉→", ""),
    "PREVIOUS TAB": ("", "←⧉", ""),
    "BACK": ("", "←", ""),
    "FORWARD": ("", "→", ""),
    "SCROLL UP": ("", "↑", ""),
    "SCROLL DOWN": ("", "↓", ""),
}
_UNKNOWN_ICON: Icon = ("", "!", "")  # warning triangle
_HAND_DOT = "●"  # filled dot, colored by hand presence
_PAUSE_ICON: Icon = ("", "⏸", "")
_PLAY_ICON: Icon = ("", "▶", "")
_CAMERA_ICON: Icon = ("", "◉", "")
_CLOSE_ICON: Icon = ("", "✕", "")


def _make_dpi_aware() -> None:
    """Have Windows draw the overlay at its real size from the start.

    PyAutoGUI marks the process DPI-aware when it is first imported, which
    happens on the first gesture. Before that, Windows scales a DPI-unaware
    window up as a bitmap, so the overlay would start large and blurry and
    shrink once a hand is seen. Doing the same call first avoids that jump.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:  # cosmetic; the overlay works without it
        pass


class StatusOverlay:
    """Small, borderless, always-on-top Tk window showing status as icons.

    One row: the current mode icon, the last action, and a dot that is green
    when a hand is seen. Drag it anywhere with the mouse. The pause/play
    button turns gesture actions off and on, the camera button shows or
    hides the camera preview (dimmed while hidden), and the cross quits
    Wave Command.
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

    def __init__(self) -> None:
        import tkinter as tk
        from tkinter import font as tkfont

        self._tk = tk
        self.quit_requested = False
        self._camera_toggle_requested = False
        self._bypass_toggle_requested = False
        self._last_status: OverlayStatus | None = None
        self._drag_offset = (0, 0)

        _make_dpi_aware()
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

        families = set(tkfont.families(root))
        icon_family = next((name for name in _ICON_FONTS if name in families), None)
        self._use_icon_font = icon_family is not None
        family = icon_family or "Segoe UI Symbol"
        badge_font = ("Segoe UI", 7, "bold")
        button_font = (family, 10)

        self._accent = tk.Frame(root, width=4, bg=self._BORDER)
        self._accent.pack(side="left", fill="y")
        content = tk.Frame(root, bg=self._BACKGROUND, padx=6, pady=3)
        content.pack(side="left")

        # Fixed widths keep the window from jumping as the icons change.
        self._mode = self._make_label(content, (family, 16), self._MUTED)
        self._mode.configure(width=2, anchor="center")
        self._mode.pack(side="left")
        self._mode_badge = self._make_label(content, badge_font, self._MUTED)
        self._mode_badge.configure(width=1)
        self._mode_badge.pack(side="left", anchor="s", pady=(0, 4))
        self._action_chip = tk.Frame(content, bg=self._BACKGROUND)
        self._action_chip.pack(side="left", padx=(4, 0))
        self._action = self._make_label(
            self._action_chip, (family, 12), self._ACTION_TEXT
        )
        self._action.configure(width=3, anchor="center", pady=2)
        self._action.pack(side="left")
        self._action_badge = self._make_label(
            self._action_chip, badge_font, self._ACTION_TEXT
        )
        self._action_badge.configure(width=1, padx=0)
        self._action_badge.pack(side="left", fill="y")
        self._hand = self._make_label(content, ("Segoe UI", 9), self._MUTED)
        self._hand.configure(text=_HAND_DOT)
        self._hand.pack(side="left", padx=(6, 8))

        self._bypass_button = self._make_button(
            content, button_font, self._request_bypass_toggle
        )
        self._camera_button = self._make_button(
            content, button_font, self._request_camera_toggle
        )
        close_button = self._make_button(
            content, button_font, lambda _event: self._request_quit()
        )
        close_button.configure(text=self._icon(_CLOSE_ICON)[0])
        # Fill every icon before measuring, so the first status does not
        # resize the window.
        self._mode.configure(text=self._icon(_MODE_ICONS["NO HAND"])[0])
        self._bypass_button.configure(text=self._icon(_PAUSE_ICON)[0])
        self._camera_button.configure(text=self._icon(_CAMERA_ICON)[0])

        for widget in (root, self._accent, content, self._mode,
                       self._mode_badge, self._action_chip, self._action,
                       self._action_badge, self._hand):
            widget.bind("<ButtonPress-1>", self._start_drag, add="+")
            widget.bind("<B1-Motion>", self._drag, add="+")

        root.update_idletasks()
        width = root.winfo_reqwidth()
        x = (root.winfo_screenwidth() - width) // 2  # centered at the top
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
        glyph, badge = self._icon(_MODE_ICONS.get(status.label, _UNKNOWN_ICON))
        self._mode.configure(text=glyph, fg=color)
        self._mode_badge.configure(text=badge, fg=color)
        self._hand.configure(fg=bgr_to_hex(status.hand_color))
        if status.action:
            glyph, badge = self._icon(
                _ACTION_ICONS.get(status.action, _UNKNOWN_ICON)
            )
            background = self._ACTION_BACKGROUND
        else:
            glyph, badge, background = "", "", self._BACKGROUND
        self._action_chip.configure(bg=background)
        self._action.configure(text=glyph, bg=background)
        self._action_badge.configure(text=badge, bg=background)
        self._camera_button.configure(
            text=self._icon(_CAMERA_ICON)[0],
            fg=self._LINK if status.camera_visible else self._MUTED,
        )
        if status.bypassed:
            self._bypass_button.configure(
                text=self._icon(_PLAY_ICON)[0], fg=self._BYPASS_ACTIVE
            )
        else:
            self._bypass_button.configure(
                text=self._icon(_PAUSE_ICON)[0], fg=self._LINK
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
        """Return True once after the camera button was clicked."""
        requested = self._camera_toggle_requested
        self._camera_toggle_requested = False
        return requested

    def consume_bypass_toggle(self) -> bool:
        """Return True once after the pause/play button was clicked."""
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

    def _make_button(self, parent: Any, font: tuple, command: Any) -> Any:
        button = self._make_label(parent, font, self._LINK)
        button.configure(cursor="hand2", padx=3)
        button.pack(side="left")
        button.bind("<Button-1>", command)
        return button

    def _icon(self, icon: Icon) -> tuple[str, str]:
        """Return (glyph, badge): the icon-font glyph, or the Unicode
        fallback (which needs no badge) when no icon font is installed."""
        if self._use_icon_font:
            return icon[0], icon[2]
        return icon[1], ""

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
