"""
Data Collection Script for Hand Gesture Recognition
===================================================

Best Practices for >95% Accuracy:
1. Collect 300-500 samples per gesture (minimum)
2. Vary hand position: near, medium, far from camera
3. Vary hand rotation: slight roll, pitch, yaw variations
4. Vary lighting conditions if possible
5. Collect both left and right hand (or mirror one)
6. Include edge cases: partial occlusions, fast movements
7. Ensure gestures are visually distinct
8. Collect data over multiple sessions

This script provides:
- Real-time feedback on collection progress
- Visual indicators for each gesture
- Automatic data balancing warnings
- Export to numpy format for training
"""

import warnings
# Suppress protobuf deprecation warning from MediaPipe dependency (floods terminal otherwise)
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import cv2
import mediapipe as mp
import numpy as np
import os
from collections import defaultdict
from datetime import datetime

# Use the standard MediaPipe API (requires the official mediapipe package)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define your gestures - customize these names and indices
GESTURE_NAMES = {
    0: "PALM",
    1: "THUMBS_RIGHT",
    2: "THUMBS_LEFT",
    3: "TWO_FINGERS_UP",
    4: "TWO_FINGERS_DOWN",
    5: "OK",
}

# Derived values so the script adapts to how many gestures you actually use
VALID_GESTURE_INDICES = sorted(GESTURE_NAMES.keys())
MAX_GESTURE_INDEX = max(VALID_GESTURE_INDICES)

# Target samples per gesture for >95% accuracy
TARGET_SAMPLES_PER_GESTURE = 400
MIN_SAMPLES_PER_GESTURE = 300

def normalize_landmarks(landmarks, mirror=False):
    """
    Normalize hand landmarks for consistent representation.
    - Translate wrist to origin
    - Scale by hand size (wrist to middle finger MCP)
    - Optionally mirror (flip x-coordinates) for left/right hand variation
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    
    # Mirror x-coordinates if requested (flip horizontally)
    if mirror:
        pts[:, 0] = 1.0 - pts[:, 0]  # Flip x: x' = 1 - x
    
    # Translate wrist (landmark 0) to origin
    wrist = pts[0].copy()
    pts -= wrist
    
    # Scale by hand size: distance from wrist to middle finger MCP (landmark 9)
    scale = np.linalg.norm(pts[9])
    if scale < 1e-6:
        scale = 1.0
    
    pts /= scale
    return pts.flatten()  # Returns 63-dim vector (21 landmarks * 3 coords)

def save_dataset(X, y, filename="hand_gestures_data.npz"):
    """Save collected dataset with metadata."""
    np.savez(
        filename,
        X=np.array(X, dtype=np.float32),
        y=np.array(y, dtype=np.int64),
        gesture_names=np.array(list(GESTURE_NAMES.values())),
        timestamp=datetime.now().isoformat()
    )
    print(f"\n✓ Saved dataset to {filename}")
    print(f"  Total samples: {len(X)}")
    print(f"  Features per sample: {X[0].shape[0]}")

def load_existing_dataset(filename="hand_gestures_data.npz"):
    """Load existing dataset to continue collection."""
    if os.path.exists(filename):
        data = np.load(filename, allow_pickle=True)
        X_arr = data["X"]
        y_arr = data["y"]

        # Always convert to Python lists so we can safely append
        X_list = [row for row in X_arr]
        y_list = [int(label) for label in y_arr]
        return X_list, y_list
    return [], []

def main():
    print("=" * 60)
    print("HAND GESTURE DATA COLLECTION")
    print("=" * 60)
    print("\nInstructions:")
    print(f"  - Press keys 0-{MAX_GESTURE_INDEX} to label current gesture")
    print("  - Hold gesture steady for 1-2 seconds while pressing key")
    print("  - Press 'm' to toggle mirror mode (saves both original + mirrored)")
    print("  - Press 's' to save current progress")
    print("  - Press 'q' to quit and save")
    print("  - Press 'r' to reset current gesture counter")
    print("\nGesture Labels:")
    for idx, name in GESTURE_NAMES.items():
        print(f"  [{idx}] {name}")
    print("\n" + "=" * 60)
    
    # Load existing data if available
    X, y = load_existing_dataset()
    print(f"\nLoaded {len(X)} existing samples")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set lower resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Initialize MediaPipe Hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    
    # Track collection statistics
    gesture_counts = defaultdict(int)
    for label in y:
        gesture_counts[label] += 1
    
    current_gesture = None
    last_key_time = {}
    mirror_mode = False  # Toggle to save both original and mirrored versions
    
    print("\nStarting collection...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand landmarks
            results = hands.process(frame_rgb)
            
            hand_detected = False
            if results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
            
            # Display statistics
            y_offset = 30
            cv2.putText(
                frame,
                "Data Collection Mode",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            y_offset += 30
            
            # Show counts for each gesture
            for idx in VALID_GESTURE_INDICES:
                count = gesture_counts[idx]
                name = GESTURE_NAMES[idx]
                color = (0, 255, 0) if count >= TARGET_SAMPLES_PER_GESTURE else (0, 165, 255) if count >= MIN_SAMPLES_PER_GESTURE else (0, 0, 255)
                status = "✓" if count >= TARGET_SAMPLES_PER_GESTURE else "⚠" if count >= MIN_SAMPLES_PER_GESTURE else "✗"
                
                cv2.putText(
                    frame,
                    f"[{idx}] {name}: {count:3d} {status}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
                y_offset += 20
            
            # Show current gesture being collected
            if current_gesture is not None:
                cv2.putText(
                    frame,
                    f"Collecting: {GESTURE_NAMES[current_gesture]}",
                    (10, y_offset + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
            
            # Show mirror mode status
            mirror_text = "Mirror Mode: ON (saving both original + mirrored)" if mirror_mode else "Mirror Mode: OFF"
            mirror_color = (0, 255, 0) if mirror_mode else (100, 100, 100)
            cv2.putText(
                frame,
                mirror_text,
                (10, y_offset + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                mirror_color,
                1,
            )
            
            # Show hand detection status
            status_text = "Hand detected ✓" if hand_detected else "No hand detected"
            status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
            cv2.putText(
                frame,
                status_text,
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                status_color,
                2,
            )
            
            # Instructions
            cv2.putText(
                frame,
                f"0-{MAX_GESTURE_INDEX}: Label | m: Mirror | s: Save | q: Quit",
                (10, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )
            
            cv2.imshow("Gesture Collection", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('m'):
                # Toggle mirror mode
                mirror_mode = not mirror_mode
                status = "ON" if mirror_mode else "OFF"
                print(f"Mirror mode: {status} (will save both original + mirrored when ON)")
            elif key == ord('s'):
                if len(X) > 0:
                    save_dataset(X, y)
                    print("\nProgress saved!")
                else:
                    print("\nNo data to save yet.")
            elif key >= ord('0') and key <= ord('9'):
                gesture_idx = key - ord('0')
                if gesture_idx not in GESTURE_NAMES:
                    print(f"⚠ No gesture configured for index {gesture_idx}. Valid indices: {VALID_GESTURE_INDICES}")
                    continue
                
                if results.multi_hand_landmarks and hand_detected:
                    # Prevent rapid duplicate captures (debounce)
                    current_time = cv2.getTickCount()
                    if gesture_idx not in last_key_time or \
                       (current_time - last_key_time[gesture_idx]) / cv2.getTickFrequency() > 0.1:
                        
                        hand_landmarks = results.multi_hand_landmarks[0]
                        
                        # Always save original
                        feat_original = normalize_landmarks(hand_landmarks.landmark, mirror=False)
                        X.append(feat_original)
                        y.append(gesture_idx)
                        gesture_counts[gesture_idx] += 1
                        samples_saved = 1
                        
                        # If mirror mode is ON, also save mirrored version
                        if mirror_mode:
                            feat_mirrored = normalize_landmarks(hand_landmarks.landmark, mirror=True)
                            X.append(feat_mirrored)
                            y.append(gesture_idx)
                            gesture_counts[gesture_idx] += 1
                            samples_saved = 2
                        
                        current_gesture = gesture_idx
                        last_key_time[gesture_idx] = current_time
                        
                        mirror_info = " (+ mirrored)" if mirror_mode else ""
                        print(f"✓ Captured [{gesture_idx}] {GESTURE_NAMES[gesture_idx]}{mirror_info} "
                              f"(Total: {gesture_counts[gesture_idx]}, Saved: {samples_saved} sample(s))")
                else:
                    print("⚠ No hand detected. Position hand in frame.")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        
        # Final save
        if len(X) > 0:
            save_dataset(X, y)
            
            # Print final statistics
            print("\n" + "=" * 60)
            print("COLLECTION SUMMARY")
            print("=" * 60)
            total_samples = len(X)
            print(f"\nTotal samples collected: {total_samples}")
            print("\nPer-gesture breakdown:")
            for idx in VALID_GESTURE_INDICES:
                count = gesture_counts[idx]
                name = GESTURE_NAMES[idx]
                percentage = (count / total_samples * 100) if total_samples > 0 else 0
                status = "✓ SUFFICIENT" if count >= TARGET_SAMPLES_PER_GESTURE else \
                         "⚠ MINIMUM" if count >= MIN_SAMPLES_PER_GESTURE else \
                         "✗ INSUFFICIENT"
                print(f"  [{idx}] {name:15s}: {count:4d} samples ({percentage:5.1f}%) {status}")
            
            # Check for class imbalance
            counts = list(gesture_counts.values())
            if counts:
                max_count = max(counts)
                min_count = min(counts)
                imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
                
                print(f"\nClass balance:")
                print(f"  Max: {max_count}, Min: {min_count}")
                print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")
                
                if imbalance_ratio > 2.0:
                    print("  ⚠ WARNING: Significant class imbalance detected!")
                    print("    Consider collecting more samples for underrepresented gestures.")
                else:
                    print("  ✓ Class balance is acceptable")
            
            # Recommendations
            print("\n" + "=" * 60)
            print("RECOMMENDATIONS FOR >95% ACCURACY:")
            print("=" * 60)
            
            insufficient = [idx for idx in VALID_GESTURE_INDICES if gesture_counts[idx] < MIN_SAMPLES_PER_GESTURE]
            if insufficient:
                print(f"\n1. Collect more samples for gestures: {[GESTURE_NAMES[i] for i in insufficient]}")
            
            below_target = [idx for idx in VALID_GESTURE_INDICES 
                          if MIN_SAMPLES_PER_GESTURE <= gesture_counts[idx] < TARGET_SAMPLES_PER_GESTURE]
            if below_target:
                print(f"2. Aim for {TARGET_SAMPLES_PER_GESTURE} samples for: {[GESTURE_NAMES[i] for i in below_target]}")
            
            print("\n3. Ensure gestures are visually distinct:")
            print("   - Each gesture should have unique finger configurations")
            print("   - Avoid gestures that look similar (e.g., FIST vs ROCK)")
            
            print("\n4. Collect data with variation:")
            print("   - Different distances from camera (near, medium, far)")
            print("   - Slight hand rotations")
            print("   - Both left and right hands (or mirror one)")
            print("   - Multiple lighting conditions")
            
            print("\n5. After training, use confusion_matrix_analysis.py to identify:")
            print("   - Which gesture pairs are confused")
            print("   - Which gestures need more training data")
            print("   - Overall model performance breakdown")

if __name__ == "__main__":
    main()
