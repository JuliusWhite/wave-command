"""Shared configuration for recognition, translation, and webcam services."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
YOLO_HAND_MODEL_PATH = MODELS_DIR / "hand-pose.pt"

CAMERA_ID = 0
CONFIDENCE_THRESHOLD = 0.5

# Ratio of thumb-to-finger tip distance to wrist-middle-MCP distance.
PINCH_THRESHOLD = 0.45
# Required ratio difference when both pinch targets are within the threshold.
PINCH_DOMINANCE_MARGIN = 0.08
# Minimum vertical wrist displacement, normalized by hand size, before scrolling.
SCROLL_START_DISTANCE_RATIO = 0.15
# Open-palm scroll velocity is normalized to hand sizes per second.
SCROLL_DEAD_ZONE_RATIO_PER_SEC = 0.08
SCROLL_SMOOTHING_ALPHA = 0.25
SCROLL_UNITS_PER_HAND_SIZE_PER_SEC = 1500.0
SCROLL_MAX_HAND_SIZES_PER_SEC = 2.5
# Horizontal wrist displacement, in hand sizes, of an index-and-middle-finger
# swipe that switches the browser tab.
TAB_SWIPE_DISTANCE_RATIO = 1.25
HISTORY_SWIPE_DISTANCE_RATIO = 1.25
TAB_SWITCH_COOLDOWN_SEC = 1.0
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
