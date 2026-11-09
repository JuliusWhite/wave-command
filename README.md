# Wave Command

Wave Command is a hands-free computer-control project that uses a webcam to interpret hand gestures and translate them into useful actions. Its broader goal is to support interaction across the computer.

## Current iteration

To keep development focused within the available time, this iteration prioritizes a small set of actions for web browsers, one of the most commonly used applications on computers. Potential actions include scrolling pages and navigating browser tabs. The gestures and their mappings to actions have not yet been defined.

The current scope does not include mouse cursor movement. This limitation applies to this iteration and do not define the long-term scope of Wave Command.

## Technology

- **Python 3** for application logic.
- **Ultralytics YOLO** for hand detection and pose estimation.
- **OpenCV** for webcam input and visual feedback.
- **PyAutoGUI / Pynput** for issuing supported computer-control inputs.
