# Project overview

## Purpose

Wave Command explores hands-free computer interaction. A webcam provides images of a user's hand; recognition turns hand landmarks into gesture states, and a translation layer maps those states to computer-action labels.

The longer-term project may support interactions across applications and the operating system. The first iteration is intentionally narrower and focuses on common browser-oriented actions.

## First-iteration behavior

The current implementation recognizes a thumb-to-index pinch using landmark indices 4 and 8. The translator maps that gesture to `CLOSE_WINDOW` in **dry-run mode**: it prints a simulated alert and the video HUD displays it. The program does not send keyboard or mouse input and does not execute OS commands.

Mouse cursor movement is outside this iteration's scope. The pinch/action mapping is an initial implementation, not a finalized gesture vocabulary.

## Recognition backend

MediaPipe Hands is the ready-to-use backend. Its pretrained hand-landmark models are provided by the installed MediaPipe package; no separate model file needs to be put in this repository's `models/` folder.

The application can also use a compatible YOLO hand-pose checkpoint at `models/hand-pose.pt`. It must provide 21 hand landmarks. The setup script currently checks whether MediaPipe works and whether optional YOLO weights exist; it does **not** train or download YOLO weights. Standard COCO pose checkpoints predict 17 body landmarks and are not compatible with the 21-point hand gesture rule.

## Safety and scope

- Translation only prints labels. There is no active OS-command adapter.
- Adding a real command executor requires an explicit design, safety review, and tests.
- Recognition and visualization are kept separate from translation so each can evolve independently.
- Up to two hands can be displayed, while the initial gesture pipeline evaluates the first detected hand.
