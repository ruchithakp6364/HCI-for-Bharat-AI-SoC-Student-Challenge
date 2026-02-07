# Real-Time Gesture Recognition – How to Use

## What You Need

| Item | Notes |
|------|--------|
| **Trained model** | `gesture_classifier.joblib` (create with `python train_model.py`) |
| **Webcam** | Built-in laptop camera or USB webcam |
| **Python env** | Same venv as collection/training (`opencv-python`, `mediapipe`, etc.) |

Nothing else to connect: the script uses your default camera (camera 0).

---

## Run Real-Time Recognition

1. **Activate your venv** (if you use one):
   ```bash
   .\.venv\Scripts\Activate.ps1
   ```

2. **Start the script**:
   ```bash
   python realtime_recognition.py
   ```

3. **Use it**:
   - A window opens with the camera view.
   - Show **one hand** in frame and make a gesture.
   - The current gesture name appears at the top (e.g. `Gesture: PALM`).
   - Latency (ms) and FPS are shown below.
   - Press **`q`** to quit.

---

## How to Test

1. **Smoke test**
   - Run `python realtime_recognition.py`.
   - Confirm the window opens and you see yourself.
   - Confirm “No hand” when your hand is out of frame, and a gesture name when your hand is in frame.

2. **Per-gesture test**
   - For each of your 6 gestures (PALM, THUMBS_RIGHT, THUMBS_LEFT, TWO_FINGERS_UP, TWO_FINGERS_DOWN, OK):
     - Make that gesture clearly.
     - Check that the label on screen matches.
   - Try different distances (near / far) and slight rotations to see if it stays stable.

3. **Latency check**
   - Look at “Latency” in the window. On a typical laptop CPU it’s often 20–50 ms per frame. If it’s much higher, close other apps or lower resolution in the script.

---

## If the Camera Doesn’t Open

- **Only one app** should use the camera; close other camera apps (Zoom, Teams, browser, etc.).
- **USB webcam:** plug it in before running the script; the script uses the default camera (index 0).
- **Windows:** allow camera access for your terminal/IDE in Settings → Privacy → Camera.
- **Multiple cameras:** the script uses `cv2.VideoCapture(0)`. To use a different camera, change `0` to `1` or `2` in `realtime_recognition.py`.

---

## Using It “In Real Time”

- **Live control:** Use the shown gesture as a trigger (e.g. “when PALM is shown, do X”). You’d add your own logic in the loop where `gesture_name` is set.
- **Demo:** Point the camera at your hand and show different gestures; the label updates every frame (smoothed over the last 7 frames).
- **Integration:** Import the same `normalize_landmarks` and `GESTURE_NAMES`, load `gesture_classifier.joblib`, and run the same pipeline (MediaPipe → normalize → predict) inside your own app or script.

No extra hardware or connections: just run the script and use the webcam.
