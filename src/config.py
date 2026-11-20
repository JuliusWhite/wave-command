"""Shared configuration for recognition, translation, and webcam services."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
YOLO_HAND_MODEL_PATH = MODELS_DIR / "hand-pose.pt"

# App logo (regenerate with scripts/make_logo.py) and Windows taskbar identity.
ASSETS_DIR = PROJECT_ROOT / "assets"
APP_ICON_PATH = ASSETS_DIR / "wave_command.ico"
APP_LOGO_PATH = ASSETS_DIR / "logo.png"
APP_USER_MODEL_ID = "WaveCommand.App"

CAMERA_ID = 0
CONFIDENCE_THRESHOLD = 0.5

# Ratio of thumb-to-finger tip distance to wrist-middle-MCP distance.
PINCH_THRESHOLD = 0.45
# Required ratio difference when both pinch targets are within the threshold.
PINCH_DOMINANCE_MARGIN = 0.08
# Consecutive frames the same pinch must be seen before it clicks, so a single
# noisy frame while the hand changes shape never clicks.
PINCH_CONFIRM_FRAMES = 2
# Once a pinch clicks it stays held until its ratio opens past this value.
# The gap to PINCH_THRESHOLD (hysteresis) stops a held pinch from flickering
# open and closed and clicking twice.
PINCH_RELEASE_THRESHOLD = 0.6
# An open palm arms one pinch for this long; stale arming does not linger.
PINCH_ARM_TIMEOUT_SEC = 3.0
# Pose stabilizer for the open-palm, index-only, two-finger, and three-finger
# poses.
# A recognized pose survives this many mismatched frames before it ends, so
# one bad frame does not reset a scroll or swipe in progress.
POSE_RELEASE_FRAMES = 3
# Frames another pose must be seen before it takes over from the current one.
POSE_SWITCH_FRAMES = 2
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
# Same for the index, middle, and ring finger swipe that switches windows.
WINDOW_SWIPE_DISTANCE_RATIO = 1.25
HISTORY_SWIPE_DISTANCE_RATIO = 1.25
# A swipe must cover its distance within this time; slow drift never swipes.
SWIPE_WINDOW_SEC = 0.8
# After a swipe, motion in the opposite direction (the hand returning) is
# ignored for this long, or until the pose is released.
SWIPE_RETURN_LOCKOUT_SEC = 1.2
TAB_SWITCH_COOLDOWN_SEC = 1.0
# Pointer (cursor) control with the pinch-ready pose.
# Frames the pose must be held before the cursor starts following the palm.
POINTER_ENTER_FRAMES = 5
# The palm must stay within this many hand sizes of where arming began; a
# moving hand (for example a swipe with a bent finger) does not start pointing.
POINTER_ARM_MAX_DRIFT_RATIO = 0.35
# Non-claw frames that pause arming instead of restarting it.
POINTER_ARM_GRACE_FRAMES = 2
# Frames without curled ring and pinky before pointing ends.
POINTER_EXIT_FRAMES = 3
# Index and middle must be bent less than this (not curled into a fist).
POINTER_MIN_FINGER_ANGLE_DEGREES = 100.0
# With depth (MediaPipe), the claw is judged by 3D finger flexion: the sum of
# the PIP and DIP joint bends. A claw bends toward the camera, which looks
# straight in the flat image, so 2D angles miss it.
# Average index/middle flexion from which the fingers count as a claw (below
# it they are straight, as in the two-finger and index-only poses).
POINTER_CLAW_MIN_FLEX_DEGREES = 35.0
# Index or middle flexion above this is a fist, not a claw.
POINTER_CLAW_MAX_FLEX_DEGREES = 150.0
# Ring or pinky flexion from which the finger counts as curled for pointing.
POINTER_CURLED_FLEX_DEGREES = 100.0
# Per-frame weight of the newest flexion value (1.0 = no smoothing).
POINTER_FLEX_SMOOTHING_ALPHA = 0.5
# The cursor freezes once a pinch ratio drops below this, before the click.
POINTER_FREEZE_PINCH_RATIO = 0.7
# One Euro filter on the palm position (image pixels): lower cutoff means
# steadier but laggier at rest; higher beta means less lag when moving fast.
POINTER_FILTER_MIN_CUTOFF_HZ = 1.0
POINTER_FILTER_BETA = 0.02
POINTER_FILTER_DERIVATIVE_CUTOFF_HZ = 1.0
# Cursor pixels per hand size of palm travel: slow (precise) to fast.
POINTER_SLOW_PX_PER_HAND_SIZE = 400.0
POINTER_FAST_PX_PER_HAND_SIZE = 1600.0
# Palm speeds (hand sizes per second) where acceleration starts and peaks.
POINTER_ACCEL_START_SPEED = 0.5
POINTER_ACCEL_FULL_SPEED = 3.0
# The camera image is not mirrored: moving the hand to the user's right moves
# it left in the image, so horizontal motion is inverted for the cursor.
POINTER_INVERT_X = True
COMMAND_COOLDOWN_SEC = 1.5
# A second left pinch within this many seconds of a left click sends a
# double-click (both clicks at once, so Windows' own double-click time and
# small hand movement between the pinches do not matter). It skips the click
# cooldown above.
DOUBLE_CLICK_WINDOW_SEC = 1.0
# During that window the second left pinch is easier to make: it counts at
# this (looser) pinch ratio, after this many frames, and needs no open palm.
# Keep the ratio below PINCH_RELEASE_THRESHOLD, or the first pinch could
# never release.
DOUBLE_CLICK_PINCH_THRESHOLD = 0.55
DOUBLE_CLICK_CONFIRM_FRAMES = 1
# With several hands in view, only the first one to appear is used. It is
# followed by palm position: a hand within this distance (normalized image
# units) of last frame's palm is the same hand. After this many frames
# without it, the lock is released and another hand can take over.
HAND_TRACK_MAX_JUMP = 0.25
HAND_TRACK_LOST_FRAMES = 8
HUD_ALERT_DURATION_SEC = 2.5
WINDOW_NAME = "Wave Command"
# Small always-on-top status window (mode, hint, hand seen, last action).
# With it running, the camera preview is optional.
SHOW_STATUS_OVERLAY = True
# Whether the camera preview opens at start; with the overlay it can be shown
# or hidden at any time. Without the overlay the preview is always shown.
SHOW_CAMERA_PREVIEW = False
OVERLAY_OPACITY = 0.9
# Distance of the overlay from the top screen edge at start (it is centered).
OVERLAY_MARGIN_PX = 20
# MediaPipe Hands / common 21-point hand layout.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)
