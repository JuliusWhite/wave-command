# Wave Command: YOLO Hand Gesture Desktop Controller 🖐️💻

AirCtrl is a real-time computer vision application that allows users to control their desktop using full-hand gestures via a standard webcam. By leveraging YOLO for fast and accurate hand-pose estimation, this project maps physical hand movements (like open palms, pinches, and swipes) to operating system commands (like scrolling, clicking, and window switching).

*Developed as a 3-week AI course project.*

## ✨ Features (Planned for v1.0)
* **Real-Time Hand Tracking:** Utilizes YOLO to detect and track full-hand keypoints with low latency.
* **Touchless Scrolling:** Move an open palm up or down to scroll web pages and documents.
* **Pinch-to-Click:** Detects thumb-to-index pinches to simulate left mouse clicks.
* **Media/Window Control:** (WIP) Swipe gestures to switch tabs or open/close fists to play/pause media.

## 🛠️ Tech Stack
* **Python 3.x** - Core programming language.
* **Ultralytics YOLOv8** - For high-speed hand detection and pose estimation.
* **OpenCV (`cv2`)** - For webcam capture and real-time visual feedback/drawing.
* **PyAutoGUI / Pynput** - For translating Python logic into actual OS-level mouse and keyboard commands.

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/AirCtrl.git](https://github.com/YourUsername/AirCtrl.git)
   cd AirCtrl