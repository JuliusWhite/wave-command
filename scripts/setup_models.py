"""Prepare and verify the configured 21-landmark hand-detection backend."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
YOLO_MODEL_PATH = MODELS_DIR / "hand-pose.pt"


def verify_mediapipe() -> None:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe is required for the built-in 21-landmark fallback. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc

    hands_api = getattr(getattr(mp, "solutions", None), "hands", None)
    if hands_api is None:
        raise RuntimeError(
            "The installed MediaPipe package does not expose the legacy Hands "
            "API required by this project. Install mediapipe>=0.10.14,<0.11.0."
        )
    hands = hands_api.Hands(static_image_mode=True, max_num_hands=1)
    hands.close()
    print("[SETUP] MediaPipe Hands is available (21 landmarks per hand).")


def verify_optional_yolo_model() -> None:
    if not YOLO_MODEL_PATH.is_file():
        print(
            "[SETUP] No custom YOLO hand model found. MediaPipe will be used. "
            "To enable YOLO, place compatible 21-keypoint weights at "
            f"{YOLO_MODEL_PATH}."
        )
        if importlib.util.find_spec("ultralytics") is None:
            print("[SETUP] Ultralytics is not installed; optional YOLO is unavailable.")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "[SETUP] Custom weights exist but Ultralytics is not installed; "
            "MediaPipe will be used."
        )
        return

    YOLO(str(YOLO_MODEL_PATH))
    print(f"[SETUP] Loaded optional YOLO model: {YOLO_MODEL_PATH}")
    print(
        "[SETUP] Ensure the model was trained for 21 hand keypoints. "
        "COCO pose weights with 17 body keypoints are not compatible."
    )


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        verify_mediapipe()
        verify_optional_yolo_model()
    except (ImportError, RuntimeError, OSError) as exc:
        print(f"[SETUP] ERROR: {exc}", file=sys.stderr)
        return 1

    print("[SETUP] Model setup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

