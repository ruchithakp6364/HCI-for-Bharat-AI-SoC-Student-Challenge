"""
Real-Time Hand Gesture Recognition
==================================

Uses your trained gesture classifier + webcam to recognize gestures live.
Same gestures and normalization as collect_gestures.py.

Requirements:
  - Trained model: gesture_classifier.joblib (run train_model.py first)
  - Webcam: built-in laptop camera or USB webcam

Usage:
  python realtime_recognition.py

Controls:
  - Show one hand in frame; gesture is shown on screen
  - Press 'q' to quit
"""

import warnings
# Suppress protobuf deprecation warning from MediaPipe dependency (floods terminal otherwise)
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import os
from collections import deque

# Must match collect_gestures.py
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

GESTURE_NAMES = {
    0: "PALM",
    1: "THUMBS_RIGHT",
    2: "THUMBS_LEFT",
    3: "TWO_FINGERS_UP",
    4: "TWO_FINGERS_DOWN",
    5: "OK",
}

# Smooth predictions over last N frames to reduce flicker
SMOOTHING_WINDOW = 7


def normalize_landmarks(landmarks):
    """Same as in collect_gestures.py (no mirror for inference)."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    wrist = pts[0].copy()
    pts -= wrist
    scale = np.linalg.norm(pts[9])
    if scale < 1e-6:
        scale = 1.0
    pts /= scale
    return pts.flatten()


def main():
    model_path = "gesture_classifier.joblib"
    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        print("Run first: python train_model.py")
        return

    print("Loading classifier...")
    clf = joblib.load(model_path)
    print("Classifier loaded.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        print("Check: 1) Camera connected 2) No other app using it 3) Permissions")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    pred_history = deque(maxlen=SMOOTHING_WINDOW)
    prev_time = time.time()

    print("\nReal-time recognition started.")
    print("Show one hand in frame. Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        start_t = time.time()

        results = hands.process(frame_rgb)
        gesture_label = None
        gesture_name = "No hand"

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
            )

            feat = normalize_landmarks(hand_landmarks.landmark)
            pred = clf.predict(feat.reshape(1, -1))[0]
            pred_history.append(pred)

            # Majority vote over recent frames
            vals, counts = np.unique(list(pred_history), return_counts=True)
            gesture_label = int(vals[np.argmax(counts)])
            gesture_name = GESTURE_NAMES.get(gesture_label, f"Label_{gesture_label}")

        end_t = time.time()
        latency_ms = (end_t - start_t) * 1000
        fps = 1.0 / (end_t - prev_time)
        prev_time = end_t

        # On-screen text
        cv2.putText(
            frame,
            f"Gesture: {gesture_name}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Latency: {latency_ms:.0f} ms | FPS: {fps:.1f}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "Press 'q' to quit",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )

        cv2.imshow("Hand Gesture Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
