import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque
import time

# ===========================
# CONFIGURATION
# ===========================
MODEL_PATH = 'gesture_classifier.joblib'

# Gesture names (must match your training)
GESTURE_NAMES = ['PALM', 'THUMBS_RIGHT', 'THUMBS_LEFT', 'TWO_FINGERS_UP', 'TWO_FINGERS_DOWN', 'OK']

print("="*60)
print("ACCURATE GESTURE RECOGNITION - DEMO MODE")
print("="*60)

# ===========================
# LOAD MODEL
# ===========================
try:
    model = joblib.load(MODEL_PATH)
    print(f"✓ Model loaded: {MODEL_PATH}")
except FileNotFoundError:
    print(f"✗ Model not found: {MODEL_PATH}")
    print("⚠️  Run 'python train_model.py' first!")
    input("Press Enter to exit...")
    exit(1)

# ===========================
# MEDIAPIPE SETUP
# ===========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
print("✓ MediaPipe initialized")

# ===========================
# NORMALIZATION FUNCTION
# ===========================
def normalize_landmarks(hand_landmarks):
    """
    Normalize hand landmarks to be translation and scale invariant.
    Returns a flattened array of 63 features (21 landmarks × 3 coordinates).
    """
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    
    # Center around wrist (landmark 0)
    wrist = landmarks[0]
    landmarks = landmarks - wrist
    
    # Scale by max distance from wrist
    distances = np.linalg.norm(landmarks, axis=1)
    max_distance = np.max(distances)
    if max_distance > 0:
        landmarks = landmarks / max_distance
    
    return landmarks.flatten()

# ===========================
# CAMERA SETUP
# ===========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("✗ Cannot open camera")
    print("⚠️  Make sure no other app is using the camera!")
    input("Press Enter to exit...")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("✓ Camera opened")

# ===========================
# SIMULATED PLAYER STATE
# ===========================
is_playing = True
is_fullscreen = False
volume = 50
current_video = 1
total_videos = 3

# ===========================
# GESTURE SMOOTHING
# ===========================
gesture_buffer = deque(maxlen=7)  # Smooth over 7 frames
last_gesture = None
gesture_cooldown = 0
current_action = ""
action_display_time = 0

# Action counter
action_counts = {name: 0 for name in GESTURE_NAMES}

print("\n" + "="*60)
print("           ACCURATE GESTURE MAPPING")
print("="*60)
print("✋ PALM             → Play/Pause Toggle")
print("                      • If playing → Pause")
print("                      • If paused → Play")
print()
print("👉 THUMBS_RIGHT    → Next Video")
print("                      • Skip to next video")
print()
print("👈 THUMBS_LEFT     → Previous Video")
print("                      • Go back to previous video")
print()
print("✌️ TWO_FINGERS_UP   → Volume UP (+10%)")
print("                      • Increase volume")
print()
print("✌️ TWO_FINGERS_DOWN → Volume DOWN (-10%)")
print("                      • Decrease volume")
print()
print("👌 OK              → Fullscreen Toggle")
print("                      • If windowed → Fullscreen")
print("                      • If fullscreen → Windowed")
print("="*60)
print("This is DEMO MODE - Actions simulated (not actual VLC)")
print("Press 'q' in the camera window to quit")
print("="*60 + "\n")

# ===========================
# MAIN LOOP
# ===========================
try:
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("✗ Failed to grab frame")
            break
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        gesture_name = "No hand"
        
        # Process hand detection
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Normalize and predict
                features = normalize_landmarks(hand_landmarks)
                features = features.reshape(1, -1)
                prediction = model.predict(features)[0]
                gesture_name = GESTURE_NAMES[prediction]
                
                # Add to buffer for smoothing
                gesture_buffer.append(gesture_name)
                
                # Get most common gesture in buffer
                if len(gesture_buffer) >= 5:
                    gesture_name = max(set(gesture_buffer), key=gesture_buffer.count)
        else:
            gesture_buffer.clear()
        
        # ===========================
        # ACCURATE GESTURE ACTIONS
        # ===========================
        gesture_cooldown -= 1
        action_display_time -= 1
        
        if gesture_name != "No hand" and gesture_name != last_gesture and gesture_cooldown <= 0:
            
            if gesture_name == "PALM":
                # PALM = Play/Pause Toggle
                is_playing = not is_playing
                state = "▶️ Playing" if is_playing else "⏸️ Paused"
                current_action = f"✋ PALM: {state}"
                action_counts[gesture_name] += 1
                print(f"{current_action} (Count: {action_counts[gesture_name]})")
                gesture_cooldown = 15
                action_display_time = 40
                
            elif gesture_name == "THUMBS_RIGHT":
                # THUMBS_RIGHT = Next Video
                current_video = (current_video % total_videos) + 1
                current_action = f"👉 THUMBS_RIGHT: Next → Video {current_video}"
                action_counts[gesture_name] += 1
                print(f"{current_action} (Count: {action_counts[gesture_name]})")
                gesture_cooldown = 20
                action_display_time = 40
                
            elif gesture_name == "THUMBS_LEFT":
                # THUMBS_LEFT = Previous Video
                current_video = current_video - 1 if current_video > 1 else total_videos
                current_action = f"👈 THUMBS_LEFT: Previous → Video {current_video}"
                action_counts[gesture_name] += 1
                print(f"{current_action} (Count: {action_counts[gesture_name]})")
                gesture_cooldown = 20
                action_display_time = 40
                
            elif gesture_name == "TWO_FINGERS_UP":
                # TWO_FINGERS_UP = Volume Up
                volume = min(volume + 10, 100)
                current_action = f"✌️ TWO_FINGERS_UP: Volume {volume}%"
                action_counts[gesture_name] += 1
                print(f"{current_action} (Count: {action_counts[gesture_name]})")
                gesture_cooldown = 8
                action_display_time = 25
                
            elif gesture_name == "TWO_FINGERS_DOWN":
                # TWO_FINGERS_DOWN = Volume Down
                volume = max(volume - 10, 0)
                current_action = f"✌️ TWO_FINGERS_DOWN: Volume {volume}%"
                action_counts[gesture_name] += 1
                print(f"{current_action} (Count: {action_counts[gesture_name]})")
                gesture_cooldown = 8
                action_display_time = 25
                
            elif gesture_name == "OK":
                # OK = Fullscreen Toggle
                is_fullscreen = not is_fullscreen
                state = "🖥️ Fullscreen ON" if is_fullscreen else "🪟 Fullscreen OFF"
                current_action = f"👌 OK: {state}"
                action_counts[gesture_name] += 1
                print(f"{current_action} (Count: {action_counts[gesture_name]})")
                gesture_cooldown = 15
                action_display_time = 40
            
            last_gesture = gesture_name
        
        # ===========================
        # CALCULATE FPS AND LATENCY
        # ===========================
        latency_ms = (time.time() - start_time) * 1000
        fps = 1 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
        
        # ===========================
        # DISPLAY INFO ON FRAME
        # ===========================
        # Top background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Gesture name (large and clear)
        cv2.putText(frame, f"Gesture: {gesture_name}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Current action (if active)
        if action_display_time > 0:
            cv2.putText(frame, current_action, (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Simulated video info
        video_info = f"Video: {current_video}/{total_videos}"
        cv2.putText(frame, video_info, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Player state
        play_state = "▶️ Playing" if is_playing else "⏸️ Paused"
        screen_state = "🖥️ Fullscreen" if is_fullscreen else "🪟 Windowed"
        state_info = f"{play_state} | {screen_state} | Vol: {volume}%"
        cv2.putText(frame, state_info, (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Bottom info bar
        cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, f"Latency: {latency_ms:.1f} ms | FPS: {fps:.1f}", (10, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "DEMO MODE - Simulated Actions", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        
        # Cooldown indicator
        if gesture_cooldown > 0:
            cooldown_text = f"Cooldown: {gesture_cooldown}"
            text_size = cv2.getTextSize(cooldown_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.putText(frame, cooldown_text, (w - text_size[0] - 10, h - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        # Show frame
        cv2.imshow('Accurate Gesture Recognition - Demo Mode', frame)
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n👋 Exiting...")
            break

except KeyboardInterrupt:
    print("\n👋 Interrupted by user...")
except Exception as e:
    print(f"\n✗ Error occurred: {e}")
    import traceback
    traceback.print_exc()

# ===========================
# CLEANUP & SUMMARY
# ===========================
print("\n" + "="*60)
print("SESSION SUMMARY - GESTURE ACCURACY TEST")
print("="*60)
for gesture, count in action_counts.items():
    if count > 0:
        print(f"{gesture:20s}: {count} times")
print("="*60)

print("\nCleaning up...")
cap.release()
cv2.destroyAllWindows()
hands.close()
print("✓ Cleanup complete")
print("="*60)