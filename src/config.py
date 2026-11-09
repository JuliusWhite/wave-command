"""Shared configuration for recognition, translation, and webcam services."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
YOLO_HAND_MODEL_PATH = MODELS_DIR / "hand-pose.pt"

CAMERA_ID = 0
CONFIDENCE_THRESHOLD = 0.5

# Ratio of thumb-index tip distance to wrist-middle-MCP distance.
PINCH_THRESHOLD = 0.35
COMMAND_COOLDOWN_SEC = 1.5
HUD_ALERT_DURATION_SEC = 2.5
WINDOW_NAME = "Wave Command"

# MediaPipe Hands / common 21-point hand layout.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

