# Project overview

## Purpose

Wave Command explores hands-free computer interaction. A webcam provides images of a user's hand; recognition turns hand landmarks into gesture states, and a translation layer maps those states to computer-action labels.

The longer-term project may support interactions across applications and the operating system. The first iteration is intentionally narrower and focuses on common browser-oriented actions.

## First-iteration behavior

The current implementation recognizes a pinch-ready claw pose for moving the cursor, thumb-to-index and thumb-to-middle pinches for left/right clicks, upright open-palm vertical movements for scrolling, horizontal two-finger (index and middle) swipes for switching tabs, and horizontal index-only swipes for back/forward browser history. The close-window gesture is temporarily disabled, with its code retained in comments for future reactivation.

Each action uses its own hand pose. An earlier version shared the open palm between scrolling (vertical) and tab switching (horizontal); the two were easily confused, so tab switching moved to the two-finger pose.

An earlier cursor attempt followed the index fingertip and was reverted: the fingertip jitters, and pinching moved it off the target. The current pointer follows the palm centre with adaptive smoothing and freezes while a pinch forms.

Cursor control is suited to links, buttons, and other normal-sized targets; very small targets remain hard. The pose/action mapping is an initial implementation, not a finalized gesture vocabulary.

## Recognition backend

MediaPipe Hands is the ready-to-use backend. Its pretrained hand-landmark models are provided by the installed MediaPipe package; no separate model file needs to be put in this repository's `models/` folder.

The application can also use a compatible YOLO hand-pose checkpoint at `models/hand-pose.pt`. It must provide 21 hand landmarks. The setup script currently checks whether MediaPipe works and whether optional YOLO weights exist; it does **not** train or download YOLO weights. Standard COCO pose checkpoints predict 17 body landmarks and are not compatible with the 21-point hand gesture rule.

## Safety and scope

- Cursor movement, pinches, scrolling, and tab/history swipes send real mouse and keyboard input. Moving the physical mouse into a screen corner aborts the app (PyAutoGUI fail-safe). The status overlay's **bypass** button (or **B** in the camera preview) turns all gesture actions off on demand without quitting.
- Cooldowns, pinch arming and confirmation, a pose stabilizer, and return-stroke lockouts limit accidental, confused, or repeated actions.
- Translation is separated from recognition so click dispatch can be tested independently.
- Recognition and visualization are kept separate from translation so each can evolve independently.
- Up to two hands can be displayed, while the initial gesture pipeline evaluates the first detected hand.
