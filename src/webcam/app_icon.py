"""Give Wave Command its own logo in the taskbar and on its windows.

On Windows, Python apps otherwise show the Python icon and are grouped with
other Python programs. Everything here is cosmetic: failures are ignored.
"""

from __future__ import annotations

import sys

from ..config import APP_ICON_PATH, APP_LOGO_PATH, APP_USER_MODEL_ID

# OpenCV's Win32 window class, so the preview is not confused with the
# status overlay, which has the same title.
_OPENCV_WINDOW_CLASS = "Main HighGUI class"
_icon_handles: tuple[int, int] | None = None


def set_app_identity() -> None:
    """Group Wave Command's windows under its own taskbar entry and icon."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            APP_USER_MODEL_ID
        )
    except Exception:
        pass


def apply_tk_icon(root: object) -> None:
    """Use the logo for a Tk root window and the dialogs it opens."""
    try:
        if sys.platform == "win32":
            root.iconbitmap(default=str(APP_ICON_PATH))
        else:
            import tkinter as tk

            image = tk.PhotoImage(master=root, file=str(APP_LOGO_PATH))
            root.iconphoto(True, image)
            root._wave_command_icon = image  # keep a reference alive
    except Exception:
        pass


def apply_opencv_icon(window_name: str) -> None:
    """Use the logo for an open OpenCV window (Windows only)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        user32.FindWindowW.restype = wintypes.HWND
        user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
        user32.SendMessageW.argtypes = [
            wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
        ]
        hwnd = user32.FindWindowW(_OPENCV_WINDOW_CLASS, window_name)
        if not hwnd:
            return
        big, small = _load_icons()
        WM_SETICON, ICON_SMALL, ICON_BIG = 0x0080, 0, 1
        if big:
            user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, big)
        if small:
            user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, small)
    except Exception:
        pass


def _load_icons() -> tuple[int, int]:
    """Load the .ico at the system's large and small icon sizes, once."""
    global _icon_handles
    if _icon_handles is None:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        user32.LoadImageW.restype = wintypes.HANDLE
        user32.LoadImageW.argtypes = [
            wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT,
            ctypes.c_int, ctypes.c_int, wintypes.UINT,
        ]
        IMAGE_ICON, LR_LOADFROMFILE = 1, 0x0010
        SM_CXICON, SM_CXSMICON = 11, 49

        def load(metric: int) -> int:
            size = user32.GetSystemMetrics(metric)
            return user32.LoadImageW(
                None, str(APP_ICON_PATH), IMAGE_ICON, size, size,
                LR_LOADFROMFILE,
            ) or 0

        _icon_handles = (load(SM_CXICON), load(SM_CXSMICON))
    return _icon_handles
