import sys
import os

print("="*60)
print("DIAGNOSTIC CHECK")
print("="*60)

# 1. Check Python version
print(f"\n1. Python version: {sys.version}")

# 2. Check required packages
print("\n2. Checking packages...")
packages = {
    'cv2': 'opencv-python',
    'mediapipe': 'mediapipe',
    'joblib': 'joblib',
    'numpy': 'numpy',
    'vlc': 'python-vlc'
}

for module, package in packages.items():
    try:
        __import__(module)
        print(f"   ✓ {package}")
    except ImportError:
        print(f"   ✗ {package} - NOT INSTALLED")
        print(f"     → Install: pip install {package}")

# 3. Check files
print("\n3. Checking files...")
required_files = [
    'gesture_classifier.joblib',
    'realtime_recognition_vlc.py',
    'train_model.py'
]

for file in required_files:
    if os.path.exists(file):
        print(f"   ✓ {file}")
    else:
        print(f"   ✗ {file} - NOT FOUND")

# 4. Check demo_videos folder
print("\n4. Checking demo_videos folder...")
VIDEOS_FOLDER = 'demo_videos'

if os.path.exists(VIDEOS_FOLDER):
    print(f"   ✓ Folder exists: {VIDEOS_FOLDER}/")
    
    files = os.listdir(VIDEOS_FOLDER)
    if files:
        print(f"   → Contents ({len(files)} files):")
        for f in files:
            path = os.path.join(VIDEOS_FOLDER, f)
            if os.path.isfile(path):
                size = os.path.getsize(path) / (1024*1024)
                print(f"      • {f} ({size:.2f} MB)")
    else:
        print(f"   ✗ Folder is EMPTY - add .mp4 files!")
else:
    print(f"   ✗ Folder does NOT exist: {VIDEOS_FOLDER}/")
    print(f"   → Create it: mkdir {VIDEOS_FOLDER}")

# 5. Test camera
print("\n5. Testing camera...")
try:
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"   ✓ Camera works! Frame size: {frame.shape}")
        else:
            print(f"   ✗ Camera opened but can't read frames")
        cap.release()
    else:
        print(f"   ✗ Cannot open camera (may be in use by another app)")
except Exception as e:
    print(f"   ✗ Camera error: {e}")

# 6. Test VLC
print("\n6. Testing VLC...")
try:
    import vlc
    print(f"   ✓ python-vlc imported")
    try:
        version = vlc.libvlc_get_version().decode()
        print(f"   → VLC version: {version}")
    except:
        print(f"   → VLC library loaded (version check failed but OK)")
except Exception as e:
    print(f"   ✗ VLC error: {e}")

# 7. Check if model is trained
print("\n7. Checking trained model...")
if os.path.exists('gesture_classifier.joblib'):
    try:
        import joblib
        model = joblib.load('gesture_classifier.joblib')
        print(f"   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Model file exists but can't load: {e}")
else:
    print(f"   ✗ Model not found - Run: python train_model.py")

print("\n" + "="*60)
print("DIAGNOSIS COMPLETE")
print("="*60)
print("\n📋 NEXT STEPS:")

if not os.path.exists('gesture_classifier.joblib'):
    print("   1. Train model first: python train_model.py")
    
if not os.path.exists('demo_videos'):
    print("   2. Create videos folder: mkdir demo_videos")
    print("   3. Add 3 .mp4 files to demo_videos/")
elif not os.listdir('demo_videos'):
    print("   2. Add 3 .mp4 files to demo_videos/")
    
print("   4. Run: python realtime_recognition_vlc.py")
print("="*60)