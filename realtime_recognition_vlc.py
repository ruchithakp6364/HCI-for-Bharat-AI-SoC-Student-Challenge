import warnings

import cv2
import mediapipe as mp
import joblib
import numpy as np
import vlc
import os
from collections import deque
import time

# Suppress protobuf deprecation warnings from MediaPipe (noise in terminal)
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# ===========================
# CONFIGURATION
# ===========================
MODEL_PATH = 'gesture_classifier.joblib'

# Video playlist - matches your folder name
VIDEOS_FOLDER = 'demo_videos'
playlist = [
    os.path.join(VIDEOS_FOLDER, 'video1.mp4'),
    os.path.join(VIDEOS_FOLDER, 'video2.mp4'),
    os.path.join(VIDEOS_FOLDER, 'video3.mp4')
]

# Gesture names (must match your training)
GESTURE_NAMES = ['PALM', 'THUMBS_RIGHT', 'THUMBS_LEFT', 'TWO_FINGERS_UP', 'TWO_FINGERS_DOWN', 'OK']

print("="*60)
print("FIXED GESTURE CONTROL FOR VLC")
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
# CHECK VIDEOS EXIST
# ===========================
if not os.path.exists(VIDEOS_FOLDER):
    print(f"✗ Videos folder not found: {VIDEOS_FOLDER}")
    print(f"⚠️  Please create a '{VIDEOS_FOLDER}' folder and add your video files!")
    input("Press Enter to exit...")
    exit(1)

missing_videos = []
for video in playlist:
    if not os.path.exists(video):
        missing_videos.append(video)

if missing_videos:
    print(f"✗ Missing videos:")
    for v in missing_videos:
        print(f"   - {v}")
    print(f"⚠️  Please add video files to the '{VIDEOS_FOLDER}' folder!")
    input("Press Enter to exit...")
    exit(1)

print(f"✓ Found {len(playlist)} videos in playlist:")
for i, v in enumerate(playlist, 1):
    size = os.path.getsize(v) / (1024*1024)
    print(f"   {i}. {os.path.basename(v)} ({size:.1f} MB)")

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
    Normalize landmarks in the SAME way as used for training
    (see collect_gestures.py & realtime_recognition.py):
      - translate wrist (landmark 0) to origin
      - scale by distance to middle-finger MCP (landmark 9)
    This keeps inference features consistent with the saved dataset.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)

    # Translate wrist to origin
    wrist = pts[0].copy()
    pts -= wrist

    # Scale by hand size: distance from wrist to middle finger MCP (landmark 9)
    scale = np.linalg.norm(pts[9])
    if scale < 1e-6:
        scale = 1.0

    pts /= scale
    return pts.flatten()

# ===========================
# VLC PLAYER SETUP
# ===========================
current_video_index = 0

try:
    print(f"✓ Initializing VLC player...")
    player = vlc.MediaPlayer(playlist[current_video_index])
    player.play()
    time.sleep(1)  # Give VLC time to start

    # Ensure a sane starting volume (some systems start at 0 or very low)
    default_volume = 60
    player.audio_set_volume(default_volume)
    print(f"✓ VLC player started at {default_volume}% volume")
    print(f"▶️  Now playing: {os.path.basename(playlist[current_video_index])}")
except Exception as e:
    print(f"✗ VLC error: {e}")
    print("⚠️  Make sure VLC media player is installed on your system!")
    input("Press Enter to exit...")
    exit(1)

# Track player state
is_playing = True
is_fullscreen = False

# ===========================
# PLAYLIST MANAGEMENT
# ===========================
def play_next_video():
    """Play the next video in the playlist"""
    global current_video_index, is_playing
    current_video_index = (current_video_index + 1) % len(playlist)
    player.stop()
    player.set_mrl(playlist[current_video_index])
    player.play()
    is_playing = True
    video_name = os.path.basename(playlist[current_video_index])
    print(f"⏭️  Next Video: {video_name} ({current_video_index + 1}/{len(playlist)})")
    return video_name

def play_previous_video():
    """Play the previous video in the playlist"""
    global current_video_index, is_playing
    current_video_index = (current_video_index - 1) % len(playlist)
    player.stop()
    player.set_mrl(playlist[current_video_index])
    player.play()
    is_playing = True
    video_name = os.path.basename(playlist[current_video_index])
    print(f"⏮️  Previous Video: {video_name} ({current_video_index + 1}/{len(playlist)})")
    return video_name

def toggle_play_pause():
    """Toggle between play and pause"""
    global is_playing
    player.pause()
    is_playing = not is_playing
    state = "▶️ Playing" if is_playing else "⏸️ Paused"
    print(f"PALM: {state}")
    return state

def _safe_get_volume():
    """Return a valid current volume (0–100), falling back to default if VLC reports an error."""
    vol = player.audio_get_volume()
    if vol < 0:  # -1 means 'error' or 'no audio'
        vol = 60
        player.audio_set_volume(vol)
    return vol

def volume_up():
    """Increase volume by a noticeable step"""
    current_vol = _safe_get_volume()
    step = 20  # bigger step so the change is clearly audible
    new_vol = min(current_vol + step, 100)
    player.audio_set_volume(new_vol)
    print(f"🔊 Volume UP: {current_vol}% → {new_vol}%")
    return new_vol

def volume_down():
    """Decrease volume by a noticeable step"""
    current_vol = _safe_get_volume()
    step = 20
    new_vol = max(current_vol - step, 0)
    player.audio_set_volume(new_vol)
    print(f"🔉 Volume DOWN: {current_vol}% → {new_vol}%")
    return new_vol

def toggle_fullscreen():
    """Toggle fullscreen mode"""
    global is_fullscreen
    player.toggle_fullscreen()
    is_fullscreen = not is_fullscreen
    state = "🖥️ Fullscreen ON" if is_fullscreen else "🪟 Fullscreen OFF (Windowed)"
    print(f"OK: {state}")
    return state

# ===========================
# CAMERA SETUP
# ===========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("✗ Cannot open camera")
    print("⚠️  Make sure no other app is using the camera!")
    player.stop()
    input("Press Enter to exit...")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("✓ Camera opened")

# ===========================
# GESTURE SMOOTHING & COOLDOWN
# ===========================
# Buffer size and stability requirements:
#  - we require several consecutive frames agreeing on the same gesture
#  - and a minimum visible hand size, so tiny/partial hands are ignored
gesture_buffer = deque(maxlen=10)  # Smooth over last 10 frames
MIN_STABLE_FRAMES = 6             # Need at least this many agreeing frames

last_executed_gesture = None      # Last gesture that triggered an action
gesture_cooldown = 0
current_action = ""
action_display_time = 0

# Track when hand disappears to reset gesture detection
frames_without_hand = 0
HAND_RESET_THRESHOLD = 10  # Reset after 10 frames without any valid hand

# Minimum hand bounding-box area (in normalized 0–1 coordinates) to accept a gesture.
# This rejects tiny/partial hands at the edge of the frame.
MIN_HAND_AREA = 0.04  # ~4% of the frame

print("\n" + "="*60)
print("           FIXED GESTURE CONTROLS")
print("="*60)
print("✋ PALM             → Play/Pause (Toggle)")
print("👉 THUMBS_RIGHT    → Next Video (repeatable)")
print("👈 THUMBS_LEFT     → Previous Video (repeatable)")
print("✌️ TWO_FINGERS_UP   → Volume UP (+10%)")
print("✌️ TWO_FINGERS_DOWN → Volume DOWN (-10%)")
print("👌 OK              → Fullscreen Toggle")
print("="*60)
print("TIP: Remove hand briefly between gestures for best results")
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
            valid_hand_detected = False

            for hand_landmarks in results.multi_hand_landmarks:
                # Compute hand bounding box in normalized coordinates (0–1)
                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]
                bbox_w = max(xs) - min(xs)
                bbox_h = max(ys) - min(ys)
                hand_area = bbox_w * bbox_h

                # If the visible hand area is too small, ignore it (likely partial / far away)
                if hand_area < MIN_HAND_AREA:
                    continue

                valid_hand_detected = True

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

            if valid_hand_detected:
                frames_without_hand = 0  # Reset counter

                # Require strong agreement in the buffer before accepting a gesture
                if len(gesture_buffer) >= MIN_STABLE_FRAMES:
                    # Most common gesture in buffer and its count
                    candidates = set(gesture_buffer)
                    majority_gesture = max(candidates, key=gesture_buffer.count)
                    majority_count = gesture_buffer.count(majority_gesture)

                    if majority_count >= MIN_STABLE_FRAMES:
                        gesture_name = majority_gesture
                    else:
                        # Not stable enough yet – treat as no gesture
                        gesture_name = "No hand"
                else:
                    # Buffer not yet full enough – treat as no gesture
                    gesture_name = "No hand"
            else:
                # Only tiny/partial hands seen – treat as no hand
                gesture_buffer.clear()
                frames_without_hand += 1
        else:
            # No hand landmarks at all
            gesture_buffer.clear()
            frames_without_hand += 1

        # Reset last executed gesture after the hand has been away for some frames
        if frames_without_hand >= HAND_RESET_THRESHOLD:
            if last_executed_gesture is not None:
                last_executed_gesture = None
                # Visual indicator that gesture is reset
                current_action = "✅ Ready for next gesture"
                action_display_time = 20
        
        # ===========================
        # GESTURE ACTIONS - FIXED LOGIC
        # ===========================
        gesture_cooldown -= 1
        action_display_time -= 1
        
        # Only trigger action if:
        # 1. A hand is detected
        # 2. Not in cooldown
        # 3. This gesture is different from the last executed gesture
        if (gesture_name != "No hand" and 
            gesture_cooldown <= 0 and 
            gesture_name != last_executed_gesture):
            
            if gesture_name == "PALM":
                # PALM = Play/Pause Toggle
                state = toggle_play_pause()
                current_action = f"✋ PALM: {state}"
                gesture_cooldown = 15
                action_display_time = 40
                last_executed_gesture = gesture_name
                
            elif gesture_name == "THUMBS_RIGHT":
                # THUMBS_RIGHT = Next Video (NO LONGER BLOCKED)
                video_name = play_next_video()
                current_action = f"👉 NEXT: {video_name}"
                gesture_cooldown = 12  # Shorter cooldown for navigation
                action_display_time = 40
                last_executed_gesture = gesture_name
                
            elif gesture_name == "THUMBS_LEFT":
                # THUMBS_LEFT = Previous Video (NO LONGER BLOCKED)
                video_name = play_previous_video()
                current_action = f"👈 PREVIOUS: {video_name}"
                gesture_cooldown = 12  # Shorter cooldown for navigation
                action_display_time = 40
                last_executed_gesture = gesture_name
                
            elif gesture_name == "TWO_FINGERS_UP":
                # TWO_FINGERS_UP = Volume Up
                new_vol = volume_up()
                current_action = f"✌️ VOLUME UP: {new_vol}%"
                gesture_cooldown = 6  # Short cooldown for volume
                action_display_time = 25
                last_executed_gesture = gesture_name
                
            elif gesture_name == "TWO_FINGERS_DOWN":
                # TWO_FINGERS_DOWN = Volume Down
                new_vol = volume_down()
                current_action = f"✌️ VOLUME DOWN: {new_vol}%"
                gesture_cooldown = 6  # Short cooldown for volume
                action_display_time = 25
                last_executed_gesture = gesture_name
                
            elif gesture_name == "OK":
                # OK = Fullscreen Toggle
                state = toggle_fullscreen()
                current_action = f"👌 OK: {state}"
                gesture_cooldown = 15
                action_display_time = 40
                last_executed_gesture = gesture_name
        
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
        cv2.rectangle(overlay, (0, 0), (w, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Gesture name (large and clear)
        cv2.putText(frame, f"Gesture: {gesture_name}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Current action (if active)
        if action_display_time > 0:
            cv2.putText(frame, current_action, (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Video info
        video_info = f"Video: {os.path.basename(playlist[current_video_index])} ({current_video_index + 1}/{len(playlist)})"
        cv2.putText(frame, video_info, (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Player state
        play_state = "▶️ Playing" if is_playing else "⏸️ Paused"
        screen_state = "🖥️ Full" if is_fullscreen else "🪟 Window"
        current_vol = player.audio_get_volume()
        state_info = f"{play_state} | {screen_state} | Vol: {current_vol}%"
        cv2.putText(frame, state_info, (10, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Gesture status indicator
        if last_executed_gesture is None:
            status_text = "Status: Ready ✓"
            status_color = (0, 255, 0)
        else:
            status_text = f"Last: {last_executed_gesture}"
            status_color = (100, 200, 255)
        cv2.putText(frame, status_text, (10, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)
        
        # Bottom info bar
        cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, f"Latency: {latency_ms:.1f}ms | FPS: {fps:.1f}", (10, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Instruction
        instruction = "Remove hand briefly to repeat same gesture"
        cv2.putText(frame, instruction, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 255), 1)
        
        # Cooldown indicator
        if gesture_cooldown > 0:
            cooldown_text = f"Cooldown: {gesture_cooldown}"
            text_size = cv2.getTextSize(cooldown_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.putText(frame, cooldown_text, (w - text_size[0] - 10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        # Show frame
        cv2.imshow('Fixed Gesture Control - VLC', frame)
        
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
# CLEANUP
# ===========================
print("\nCleaning up...")
cap.release()
cv2.destroyAllWindows()
hands.close()
player.stop()
print("✓ Cleanup complete")
print("="*60)