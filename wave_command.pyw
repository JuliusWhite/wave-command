"""Windowless launcher: run Wave Command without a terminal.

Opened with ``pythonw`` (double-click, or the shortcut made by
``scripts/create_shortcut.py``), so no console window appears. Terminal
output goes to ``wave-command.log`` next to this file, and a crash shows
an error dialog instead of vanishing silently.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_PATH = PROJECT_ROOT / "wave-command.log"


def show_error(message: str) -> None:
    """Show a blocking error dialog; fall back to doing nothing."""
    if sys.platform == "win32":
        import ctypes

        MB_ICONERROR = 0x10
        ctypes.windll.user32.MessageBoxW(None, message, "Wave Command", MB_ICONERROR)
        return
    try:
        import tkinter
        from tkinter import messagebox

        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror("Wave Command", message)
        root.destroy()
    except Exception:
        pass


def run() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    with open(LOG_PATH, "w", encoding="utf-8", buffering=1) as log:
        sys.stdout = sys.stderr = log
        try:
            from main import main

            main()
        except SystemExit:
            raise
        except BaseException as error:
            traceback.print_exc()
            show_error(
                f"Wave Command stopped because of an error:\n\n{error}\n\n"
                f"Details are in:\n{LOG_PATH}"
            )
            sys.exit(1)


if __name__ == "__main__":
    run()
