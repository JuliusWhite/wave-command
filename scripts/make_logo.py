"""Draw the Wave Command logo and write it to ``assets/``.

The mark is a single electric-blue stroke that reads both as a wave and as
a "W", on the deep navy panel of the status overlay. Light dots on its three
crests stand for tracked fingertip landmarks.

    python scripts/make_logo.py

Writes ``assets/logo.png`` (512 px) and ``assets/wave_command.ico``
(16-256 px). Requires Pillow, which is installed with the base requirements.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"

# Brand palette, matching src/webcam/overlay.py.
NAVY = (11, 22, 38)  # #0b1626
BLUE = (30, 143, 255)  # #1e8fff
LIGHT_BLUE = (92, 208, 255)  # #5cd0ff
DOT = (232, 246, 255)  # #e8f6ff

CANVAS = 2048  # drawn large, then downsampled for smooth edges


def wave_point(t: float) -> tuple[float, float]:
    """Point on the W wave for t in [0, 1]: crests at 0, 0.5, and 1."""
    x = CANVAS * (0.18 + 0.64 * t)
    y = CANVAS * (0.5 - 0.11 * math.cos(4 * math.pi * t))
    return x, y


def draw_logo() -> Image.Image:
    image = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (0, 0, CANVAS - 1, CANVAS - 1), radius=int(CANVAS * 0.22), fill=NAVY
    )

    # Stroke mask: stamp round discs along the curve for round caps/joins.
    mask = Image.new("L", image.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    half_width = CANVAS * 0.055
    steps = 2000
    for i in range(steps + 1):
        x, y = wave_point(i / steps)
        mask_draw.ellipse(
            (x - half_width, y - half_width, x + half_width, y + half_width),
            fill=255,
        )

    # Left-to-right gradient from the overlay blue to the lighter accent.
    gradient = Image.new("RGBA", image.size)
    gradient_draw = ImageDraw.Draw(gradient)
    for x in range(CANVAS):
        mix = x / (CANVAS - 1)
        color = tuple(round(a + (b - a) * mix) for a, b in zip(BLUE, LIGHT_BLUE))
        gradient_draw.line((x, 0, x, CANVAS), fill=color + (255,))
    image.paste(gradient, (0, 0), mask)

    # Fingertip landmarks on the three crests, ringed in navy so they pop.
    dot_radius = CANVAS * 0.054
    ring = CANVAS * 0.02
    for t in (0.0, 0.5, 1.0):
        x, y = wave_point(t)
        draw.ellipse(
            (x - dot_radius - ring, y - dot_radius - ring,
             x + dot_radius + ring, y + dot_radius + ring),
            fill=NAVY,
        )
        draw.ellipse(
            (x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius),
            fill=DOT,
        )
    return image


def main() -> None:
    ASSETS_DIR.mkdir(exist_ok=True)
    logo = draw_logo()
    logo.resize((512, 512), Image.LANCZOS).save(ASSETS_DIR / "logo.png")
    logo.resize((256, 256), Image.LANCZOS).save(
        ASSETS_DIR / "wave_command.ico",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128),
               (256, 256)],
    )
    print(f"Wrote logo.png and wave_command.ico to {ASSETS_DIR}")


if __name__ == "__main__":
    main()
