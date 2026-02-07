import os

print("="*50)
print("CHECKING YOUR SETUP")
print("="*50)

# Check model
if os.path.exists('gesture_classifier.joblib'):
    print("✓ Model found: gesture_classifier.joblib")
else:
    print("✗ Model NOT found: gesture_classifier.joblib")
    print("  → Run: python train_model.py")

# Check videos folder
VIDEOS_FOLDER = 'demo_video'
if os.path.exists(VIDEOS_FOLDER):
    print(f"✓ Folder found: {VIDEOS_FOLDER}/")
    
    # List videos
    videos = [f for f in os.listdir(VIDEOS_FOLDER) if f.endswith('.mp4')]
    if videos:
        print(f"  → Found {len(videos)} video(s):")
        for v in videos:
            size = os.path.getsize(os.path.join(VIDEOS_FOLDER, v)) / (1024*1024)
            print(f"     • {v} ({size:.2f} MB)")
    else:
        print(f"  ✗ No .mp4 files in {VIDEOS_FOLDER}/")
else:
    print(f"✗ Folder NOT found: {VIDEOS_FOLDER}/")
    print(f"  → Create it: mkdir {VIDEOS_FOLDER}")

# Check camera
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✓ Camera is accessible")
    cap.release()
else:
    print("✗ Camera NOT accessible")

print("="*50)