"""Always-on-top status window, so the camera preview is optional.

The overlay is a small rounded card that stays above other windows and
shows the current mode, the last action, and whether a hand is seen, in the
same colors as the camera HUD. It is driven from the camera
loop: call ``show`` with the latest status and ``pump`` once per frame.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Any

from ..config import OVERLAY_MARGIN_PX, OVERLAY_OPACITY, WINDOW_NAME
from .app_icon import apply_tk_icon

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


def _hex_to_rgb(color: str) -> tuple[int, ...]:
    """Convert a Tk ``#rrggbb`` string to (red, green, blue)."""
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


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
    "WINDOWS": ("", "❐", ""),        # task view
    "BYPASSED": ("", "⏸", ""),       # pause
}
_ACTION_ICONS: dict[str, Icon] = {
    "LEFT CLICK": ("", "◧", "L"),
    "RIGHT CLICK": ("", "◨", "R"),
    "DOUBLE CLICK": ("", "◧◧", "2"),
    "NEXT TAB": ("", "⧉→", ""),
    "PREVIOUS TAB": ("", "←⧉", ""),
    "NEXT WINDOW": ("", "❐→", ""),
    "PREVIOUS WINDOW": ("", "←❐", ""),
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
_HIDE_ICON: Icon = ("", "▲", "")  # chevron up
_SHOW_ICON: Icon = ("", "▼", "")  # chevron down
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
    """Rounded, borderless, always-on-top Tk card showing the status.

    Laid out like a "now playing" card: a tile with the current mode icon on
    the left, the mode name and a hint (or the last action) next to it, a
    tile with the last action icon on the right (a white icon on blue while
    an action shows, a dim slot otherwise), and the pause/play,
    camera and close buttons in a column at the edge. A dot after the mode
    name is green when a hand is seen. Drag it anywhere with the mouse. The
    pause/play button turns gesture actions off and on, the camera button
    shows or hides the camera preview (dimmed while hidden), the cross quits
    Wave Command, and the small chevron at the bottom center shrinks the card
    to a tab (click the tab to bring it back).
    """

    # Brand palette: light, electric blue on a deep navy card.
    _GRADIENT_LEFT = "#173a63"
    _GRADIENT_RIGHT = "#0b1626"
    _TILE = "#1f4a7a"
    # Same family as the mode tile, a step brighter, so an action stands out
    # without clashing with the card's gradient.
    _ACTION_TILE = "#2a5f96"
    # A hair lighter than the card, so the empty slot almost disappears.
    _ACTION_TILE_IDLE = "#112440"
    _TEXT = "#e8f6ff"
    _MUTED = "#7ea7cc"
    _LINK = "#5cd0ff"
    _ACTION_GLYPH = "#ffffff"
    # Painted around the rounded corners and made see-through (Windows).
    _TRANSPARENT_KEY = "#010203"

    # Layout in pixels at 96 DPI; scaled to the screen's DPI.
    _HEIGHT = 76
    _PAD = 11
    _TILE_SIZE = 54
    _TEXT_WIDTH = 250
    _BUTTON_COLUMN = 38
    _RADIUS = 18
    _TILE_RADIUS = 12
    # The small tab left in place while the card is hidden.
    _TAB_WIDTH = 60
    _TAB_HEIGHT = 20
    _TAB_RADIUS = 10

    def __init__(self) -> None:
        import tkinter as tk
        from tkinter import font as tkfont

        self._tk = tk
        self.quit_requested = False
        self._camera_toggle_requested = False
        self._bypass_toggle_requested = False
        self._last_status: OverlayStatus | None = None
        self._drag_offset: tuple[int, int] | None = None
        self._hidden = False

        _make_dpi_aware()
        root = tk.Tk()
        root.title(WINDOW_NAME)
        apply_tk_icon(root)
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        try:
            root.attributes("-alpha", OVERLAY_OPACITY)
        except tk.TclError:
            pass
        key = self._TRANSPARENT_KEY
        try:
            root.attributes("-transparentcolor", key)
        except tk.TclError:  # not on this platform: corners show the card color
            key = self._GRADIENT_RIGHT
        root.configure(bg=key)
        root.protocol("WM_DELETE_WINDOW", self._request_quit)
        self._root = root

        families = set(tkfont.families(root))
        icon_family = next((name for name in _ICON_FONTS if name in families), None)
        self._use_icon_font = icon_family is not None
        family = icon_family or "Segoe UI Symbol"
        self._subtitle_font = tkfont.Font(root, family="Segoe UI", size=9)
        badge_font = ("Segoe UI", 7, "bold")

        scale = max(1.0, root.winfo_fpixels("1i") / 96)

        def s(value: float) -> int:
            return round(value * scale)

        height, pad, tile = s(self._HEIGHT), s(self._PAD), s(self._TILE_SIZE)
        self._text_width = s(self._TEXT_WIDTH)
        text_x = pad + tile + s(12)
        action_x = text_x + self._text_width + s(8)
        buttons_x = action_x + tile + s(2)
        width = buttons_x + s(self._BUTTON_COLUMN)
        tile_y = (height - tile) // 2
        self._dot_gap = s(6)

        canvas = tk.Canvas(
            root, width=width, height=height, bg=key,
            highlightthickness=0, borderwidth=0,
        )
        canvas.pack()
        self._canvas = canvas
        self._card_size = (width, height)
        self._card_columns = self._draw_card(
            canvas, width, height, s(self._RADIUS)
        )
        self._top_rounded = False

        # Left tile: the current mode.
        self._rounded_rect(pad, tile_y, pad + tile, tile_y + tile,
                           s(self._TILE_RADIUS), fill=self._TILE)
        self._mode = canvas.create_text(
            pad + tile // 2, tile_y + tile // 2, font=(family, 18),
            fill=self._MUTED,
        )
        self._mode_badge = canvas.create_text(
            pad + tile - s(5), tile_y + tile - s(4), anchor="se",
            font=badge_font, fill=self._MUTED,
        )

        # Middle: the mode name with the hand dot, and the hint or action.
        self._title = canvas.create_text(
            text_x, height // 2, anchor="sw",
            font=("Segoe UI", 11, "bold"), fill=self._TEXT,
        )
        self._hand = canvas.create_text(
            text_x, height // 2 - s(3), anchor="sw", text=_HAND_DOT,
            font=("Segoe UI", 8), fill=self._MUTED,
        )
        self._subtitle = canvas.create_text(
            text_x, height // 2 + s(1), anchor="nw",
            font=self._subtitle_font, fill=self._MUTED,
        )

        # Right tile: the last action, white on blue (a dim slot while idle).
        self._action_tile = self._rounded_rect(
            action_x, tile_y, action_x + tile, tile_y + tile,
            s(self._TILE_RADIUS), fill=self._ACTION_TILE_IDLE,
        )
        self._action = canvas.create_text(
            action_x + tile // 2, tile_y + tile // 2, font=(family, 18),
            fill=self._ACTION_GLYPH,
        )
        self._action_badge = canvas.create_text(
            action_x + tile - s(5), tile_y + tile - s(4), anchor="se",
            font=badge_font, fill=self._ACTION_GLYPH,
        )

        # Button column at the right edge.
        column_x = buttons_x + s(self._BUTTON_COLUMN) // 2 - s(3)
        button_font = (family, 10)
        self._bypass_button = self._make_button(
            column_x, height // 2 - s(21), button_font, self._request_bypass_toggle
        )
        self._camera_button = self._make_button(
            column_x, height // 2, button_font, self._request_camera_toggle
        )
        close_button = self._make_button(
            column_x, height // 2 + s(21), button_font,
            lambda _event: self._request_quit(),
        )
        # The hide chevron sits at the bottom center, where the tab's
        # chevron appears once the card is hidden.
        hide_button = self._make_button(
            width // 2, height - s(8), (family, 9), self._hide
        )
        canvas.itemconfigure(
            hide_button, text=self._icon(_HIDE_ICON)[0], fill=self._MUTED
        )
        canvas.itemconfigure(close_button, text=self._icon(_CLOSE_ICON)[0])
        canvas.itemconfigure(
            self._bypass_button, text=self._icon(_PAUSE_ICON)[0]
        )
        canvas.itemconfigure(
            self._camera_button, text=self._icon(_CAMERA_ICON)[0]
        )

        canvas.bind("<ButtonPress-1>", self._start_drag)
        canvas.bind("<B1-Motion>", self._drag)

        # The tab shown instead of the card while it is hidden; a click on
        # it anywhere shows the card again.
        tab_width, tab_height = s(self._TAB_WIDTH), s(self._TAB_HEIGHT)
        tab = tk.Canvas(
            root, width=tab_width, height=tab_height, bg=key,
            highlightthickness=0, borderwidth=0, cursor="hand2",
        )
        self._draw_card(tab, tab_width, tab_height, s(self._TAB_RADIUS))
        self._tab_glyph = tab.create_text(
            tab_width // 2, tab_height // 2, font=(family, 9),
            fill=self._LINK, text=self._icon(_SHOW_ICON)[0],
        )
        tab.bind("<ButtonPress-1>", self._show_card)
        self._tab = tab
        self._tab_size = (tab_width, tab_height)

        root.update_idletasks()
        x = (root.winfo_screenwidth() - width) // 2  # centered at the top
        self._home = (max(0, x), OVERLAY_MARGIN_PX)
        root.geometry(f"{width}x{height}+{self._home[0]}+{self._home[1]}")
        self._round_top_corners(self._home[1] > 0)
        root.update_idletasks()
        self._make_non_activating()

    def show(self, status: OverlayStatus) -> None:
        """Display a status; items are only touched when it changes."""
        if status == self._last_status:
            return
        self._last_status = status
        canvas = self._canvas
        color = bgr_to_hex(status.color)
        glyph, badge = self._icon(_MODE_ICONS.get(status.label, _UNKNOWN_ICON))
        canvas.itemconfigure(self._mode, text=glyph, fill=color)
        canvas.itemconfigure(self._mode_badge, text=badge, fill=color)
        # While bypassed, the hidden tab's chevron takes the hand dot's red.
        tab_color = bgr_to_hex(status.hand_color) if status.bypassed else color
        self._tab.itemconfigure(self._tab_glyph, fill=tab_color)

        canvas.itemconfigure(self._title, text=status.label.capitalize())
        title_right = canvas.bbox(self._title)[2]
        _, dot_y = canvas.coords(self._hand)
        canvas.coords(self._hand, title_right + self._dot_gap, dot_y)
        canvas.itemconfigure(self._hand, fill=bgr_to_hex(status.hand_color))

        if status.action:
            glyph, badge = self._icon(
                _ACTION_ICONS.get(status.action, _UNKNOWN_ICON)
            )
            subtitle, subtitle_color = status.action.capitalize(), self._LINK
            tile = self._ACTION_TILE
        else:
            glyph, badge, tile = "", "", self._ACTION_TILE_IDLE
            subtitle, subtitle_color = status.hint, self._MUTED
        canvas.itemconfigure(
            self._subtitle, text=self._fit(subtitle), fill=subtitle_color
        )
        canvas.itemconfigure(self._action_tile, fill=tile)
        canvas.itemconfigure(self._action, text=glyph)
        canvas.itemconfigure(self._action_badge, text=badge)

        canvas.itemconfigure(
            self._camera_button,
            fill=self._LINK if status.camera_visible else self._MUTED,
        )
        icon = _PLAY_ICON if status.bypassed else _PAUSE_ICON
        canvas.itemconfigure(self._bypass_button, text=self._icon(icon)[0])

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

    def _hide(self, _event: Any) -> str:
        """Swap the card for the small tab, at the top center of the screen."""
        if not self._hidden:
            self._hidden = True
            tab_width, tab_height = self._tab_size
            x = (self._root.winfo_screenwidth() - tab_width) // 2
            self._canvas.pack_forget()
            self._canvas.configure(cursor="")
            self._tab.pack()
            self._root.geometry(f"{tab_width}x{tab_height}+{max(0, x)}+0")
        return "break"

    def _show_card(self, _event: Any) -> str:
        """Bring the card back at its starting place, even if it was dragged."""
        if self._hidden:
            self._hidden = False
            width, height = self._card_size
            x, y = self._home
            self._tab.pack_forget()
            self._canvas.pack()
            self._root.geometry(f"{width}x{height}+{x}+{y}")
            self._round_top_corners(y > 0)
        return "break"

    def _round_top_corners(self, rounded: bool) -> None:
        """Round the card's top corners too, or square them again.

        Docked at the top screen edge, the card hangs from it with square top
        corners; anywhere else all four corners are rounded.
        """
        if rounded == self._top_rounded:
            return
        self._top_rounded = rounded
        _, height = self._card_size
        for item, x, inset in self._card_columns:
            top = inset if rounded else 0
            self._canvas.coords(item, x, top, x, height - inset)

    def _draw_card(
        self, canvas: Any, width: int, height: int, radius: int
    ) -> list[tuple[int, int, int]]:
        """Paint the card: a left-to-right gradient that hangs from the top
        screen edge, so only its bottom corners are rounded.

        Each one-pixel column is shortened near the left and right edges so
        that together they trace the bottom corners. Returns each column's
        (item, x, inset), so the top corners can be rounded later.
        """
        columns = []
        left = _hex_to_rgb(self._GRADIENT_LEFT)
        right = _hex_to_rgb(self._GRADIENT_RIGHT)
        for x in range(width):
            edge = min(x + 0.5, width - x - 0.5)
            inset = 0.0
            if edge < radius:
                inset = radius - math.sqrt(radius**2 - (radius - edge) ** 2)
            t = x / max(1, width - 1)
            color = "#" + "".join(
                f"{round(a + (b - a) * t):02x}" for a, b in zip(left, right)
            )
            inset_px = round(inset)
            item = canvas.create_line(x, 0, x, height - inset_px, fill=color)
            columns.append((item, x, inset_px))
        return columns

    def _rounded_rect(
        self, x1: int, y1: int, x2: int, y2: int, radius: int, **options: Any
    ) -> int:
        """Draw a filled rounded rectangle as a smoothed polygon."""
        points = (
            x1 + radius, y1, x2 - radius, y1, x2, y1, x2, y1 + radius,
            x2, y2 - radius, x2, y2, x2 - radius, y2, x1 + radius, y2,
            x1, y2, x1, y2 - radius, x1, y1 + radius, x1, y1,
        )
        return self._canvas.create_polygon(points, smooth=True, **options)

    def _make_button(self, x: int, y: int, font: tuple, command: Any) -> int:
        canvas = self._canvas
        item = canvas.create_text(
            x, y, font=font, fill=self._LINK, tags=("button",)
        )
        canvas.tag_bind(item, "<ButtonPress-1>", command)
        canvas.tag_bind(
            item, "<Enter>", lambda _event: canvas.configure(cursor="hand2")
        )
        canvas.tag_bind(
            item, "<Leave>", lambda _event: canvas.configure(cursor="")
        )
        return item

    def _fit(self, text: str) -> str:
        """Shorten text with an ellipsis so it fits the text column."""
        measure = self._subtitle_font.measure
        if measure(text) <= self._text_width:
            return text
        while text and measure(text + "…") > self._text_width:
            text = text[:-1]
        return text.rstrip() + "…"

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
        return "break"

    def _request_bypass_toggle(self, _event: Any) -> str:
        self._bypass_toggle_requested = True
        return "break"

    def _start_drag(self, event: Any) -> None:
        if "button" in self._canvas.gettags("current"):
            self._drag_offset = None  # a click on a button does not drag
            return
        self._drag_offset = (
            event.x_root - self._root.winfo_x(),
            event.y_root - self._root.winfo_y(),
        )

    def _drag(self, event: Any) -> None:
        if self._drag_offset is None:
            return
        dx, dy = self._drag_offset
        y = max(0, event.y_root - dy)  # dragged up, it docks at the edge
        self._root.geometry(f"+{event.x_root - dx}+{y}")
        self._round_top_corners(y > 0)

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
