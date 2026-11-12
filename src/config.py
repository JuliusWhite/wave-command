"""Shared configuration for recognition, translation, and webcam services."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
YOLO_HAND_MODEL_PATH = MODELS_DIR / "hand-pose.pt"

CAMERA_ID = 0
CONFIDENCE_THRESHOLD = 0.5

# Pinch ratios use wrist-to-middle-MCP distance as the hand-size reference.
PINCH_THRESHOLD = 0.30
COMMAND_COOLDOWN_SEC = 1.5
HUD_ALERT_DURATION_SEC = 2.5
WINDOW_NAME = "Wave Command"
POINTER_ACTIVE_MARGIN = 0.08
POINTER_SMOOTHING = 0.35
POINTER_DEADZONE_PX = 2.0
POINTER_MAX_SPEED_PX_PER_SEC = 600.0
POINTER_SCREEN_MARGIN_PX = 3
POINTER_MIRROR_X = False
POINTER_INDEX_EXTENSION_RATIO = 1.15
POINTER_OTHER_FINGER_CURL_RATIO = 0.85
POINTER_THUMB_CURL_RATIO = 0.85

# MediaPipe Hands / common 21-point hand layout.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)
