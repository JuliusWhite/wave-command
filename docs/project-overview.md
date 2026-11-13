# Project overview

## Purpose

Wave Command explores hands-free computer interaction. A webcam provides images of a user's hand; recognition turns hand landmarks into gesture states, and a translation layer maps those states to computer-action labels.

The longer-term project may support interactions across applications and the operating system. The first iteration is intentionally narrower and focuses on common browser-oriented actions.

## First-iteration behavior

The current implementation recognizes thumb-to-index and thumb-to-middle pinches for left/right clicks, horizontal open-hand swipes for next/previous browser tabs, and horizontal two-finger swipes for back/forward browser history. Scrolling and the close-window gesture are temporarily disabled, with their code retained in comments for future reactivation. Cursor movement is not implemented.

Position the cursor manually before pinching. The pinch/action mapping is an initial implementation, not a finalized gesture vocabulary.

## Recognition backend

MediaPipe Hands is the ready-to-use backend. Its pretrained hand-landmark models are provided by the installed MediaPipe package; no separate model file needs to be put in this repository's `models/` folder.

The application can also use a compatible YOLO hand-pose checkpoint at `models/hand-pose.pt`. It must provide 21 hand landmarks. The setup script currently checks whether MediaPipe works and whether optional YOLO weights exist; it does **not** train or download YOLO weights. Standard COCO pose checkpoints predict 17 body landmarks and are not compatible with the 21-point hand gesture rule.

## Safety and scope

- Pinches and tab/history swipes send real mouse clicks and keyboard shortcuts; cursor movement is not implemented.
- Translation is separated from recognition so click dispatch can be tested independently.
- Recognition and visualization are kept separate from translation so each can evolve independently.
- Up to two hands can be displayed, while the initial gesture pipeline evaluates the first detected hand.
