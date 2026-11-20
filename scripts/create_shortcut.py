"""Create Windows shortcuts that start Wave Command without VS Code.

Run it once with the project's virtual environment active:

    python scripts/create_shortcut.py

It adds a "Wave Command" shortcut to the Desktop and the Start Menu. The
shortcut starts ``wave_command.pyw`` with this environment's ``pythonw.exe``,
so no terminal window opens. Pass ``--no-desktop`` or ``--no-start-menu`` to
skip either location.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "wave_command.pyw"
SHORTCUT_NAME = "Wave Command.lnk"
ICON = PROJECT_ROOT / "assets" / "wave_command.ico"


def windowless_python() -> Path:
    """Return pythonw.exe for the interpreter running this script."""
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    if not pythonw.exists():
        sys.exit(f"pythonw.exe not found next to {sys.executable}")
    return pythonw


def special_folder(name: str) -> Path:
    """Resolve a Windows special folder (handles OneDrive-redirected Desktops)."""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"[Environment]::GetFolderPath('{name}')"],
        capture_output=True, text=True, check=True,
    )
    return Path(result.stdout.strip())


def create_shortcut(path: Path, target: Path) -> None:
    def quote(value: object) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    launcher_arg = f'"{LAUNCHER}"'

    script = "; ".join([
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut("
        f"{quote(path)})",
        f"$s.TargetPath = {quote(target)}",
        f"$s.Arguments = {quote(launcher_arg)}",
        f"$s.WorkingDirectory = {quote(PROJECT_ROOT)}",
        f"$s.IconLocation = {quote(ICON)}",
        "$s.Description = 'Hands-free computer control with webcam gestures'",
        "$s.Save()",
    ])
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True)
    print(f"Created {path}")


def main() -> None:
    if sys.platform != "win32":
        sys.exit("Shortcuts are only created on Windows. Elsewhere, run: "
                 f"pythonw {LAUNCHER}")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-desktop", action="store_true")
    parser.add_argument("--no-start-menu", action="store_true")
    args = parser.parse_args()

    target = windowless_python()
    if not args.no_desktop:
        create_shortcut(special_folder("Desktop") / SHORTCUT_NAME, target)
    if not args.no_start_menu:
        create_shortcut(special_folder("Programs") / SHORTCUT_NAME, target)


if __name__ == "__main__":
    main()
