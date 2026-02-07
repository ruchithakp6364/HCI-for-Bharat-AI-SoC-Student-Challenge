# Real-Time Hand Gesture Recognition System

A computer vision system for controlling VLC media player using hand gestures captured via webcam. Built with MediaPipe for hand tracking and scikit-learn for gesture classification.

## Features

- **6 Static Gestures** for media control
- **Real-time Recognition** with low latency (<50ms)
- **VLC Media Player Integration** for hands-free control
- **High Accuracy** gesture classification using Random Forest
- **Easy Training** with interactive data collection

## Demo

Control VLC media player with gestures:
- ✋ **PALM** - Play/Pause toggle
- 👉 **THUMBS_RIGHT** - Next video
- 👈 **THUMBS_LEFT** - Previous video  
- ✌️ **TWO_FINGERS_UP** - Volume up (+10%)
- ✌️ **TWO_FINGERS_DOWN** - Volume down (-10%)
- 👌 **OK** - Fullscreen toggle

## Installation

### Prerequisites

- Python 3.8+
- Webcam
- [VLC Media Player](https://www.videolan.org/vlc/) (for VLC control features)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd cursor-new
```

2. Create virtual environment:
```bash
python -m venv .venv
```

3. Activate virtual environment:
```bash
# Windows
.\.venv\Scripts\Activate.ps1

# Mac/Linux
source .venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Collect Training Data

```bash
python collect_gestures.py
```

**Instructions:**
- Follow on-screen prompts for each gesture
- Collect 150-200 samples per gesture
- Press **SPACE** to start/stop collection
- Press **Q** when finished
- Make clear, distinct gestures with good lighting

### 2. Train the Model

```bash
python train_model.py
```

Creates `gesture_classifier.joblib` with trained Random Forest model.

### 3. Run Demo Mode (No VLC Required)

```bash
python realtime_recognition_demo_accurate.py
```

Tests gesture recognition without VLC - displays simulated actions.

### 4. Run with VLC Control

```bash
python realtime_vlc_FIXED.py
```

**Setup:**
- Create `demo_videos` folder in project directory
- Add 3 video files: `video1.mp4`, `video2.mp4`, `video3.mp4`
- Run the script
- VLC will open automatically and respond to gestures

## Project Structure

```
cursor-new/
├── collect_gestures.py                      # Interactive data collection
├── train_model.py                           # Model training script
├── realtime_recognition_demo_accurate.py    # Demo mode (no VLC)
├── realtime_vlc_FIXED.py                   # VLC control with gestures
├── recollect_palm.py                        # Re-collect specific gesture
├── clean_data.py                            # Clean old training data
├── diagnose.py                              # System diagnostics
├── confusion_matrix_analysis.py             # Performance analysis
├── requirements.txt                         # Python dependencies
├── demo_videos/                             # Video files for VLC
│   ├── video1.mp4
│   ├── video2.mp4
│   └── video3.mp4
├── gesture_classifier.joblib                # Trained model (generated)
└── hand_gestures_data.npz                   # Training data (generated)
```

## Gesture Definitions

Make gestures clear and distinct for best recognition:

| Gesture | Description |
|---------|-------------|
| **PALM** | Open hand, fingers together, palm facing camera |
| **THUMBS_RIGHT** | Fist with thumb pointing right |
| **THUMBS_LEFT** | Fist with thumb pointing left |
| **TWO_FINGERS_UP** | Peace sign pointing upward (index + middle finger) |
| **TWO_FINGERS_DOWN** | Peace sign pointing downward |
| **OK** | Thumb and index finger making circle, other fingers extended |

## Usage Tips

### For Best Gesture Recognition

1. **Lighting**: Ensure even lighting on your hand
2. **Background**: Use plain, contrasting background
3. **Distance**: Keep hand 1-2 feet from camera
4. **Centering**: Keep hand in center of frame
5. **Consistency**: Make gestures the same way each time
6. **Variation**: During collection, vary angle slightly

### Repeating Gestures

To trigger the same gesture multiple times:
1. Make the gesture
2. Remove hand from frame briefly
3. Wait for "Ready" indicator
4. Make gesture again

### Improving Accuracy

If a specific gesture isn't detected well:

```bash
python recollect_palm.py  # Example for PALM gesture
```

Modify the script for other gestures, then retrain:

```bash
python train_model.py
```

## Troubleshooting

### Camera Not Opening

- Close other applications using the camera (Zoom, Teams, etc.)
- Check camera permissions in Windows Settings → Privacy → Camera
- Try different camera index (change `0` to `1` in script)

### VLC Not Starting

- Ensure VLC is installed: https://www.videolan.org/vlc/
- Install with default settings (usually `C:\Program Files\VideoLAN\VLC\`)
- Restart computer after installation

### Low Accuracy

```bash
python confusion_matrix_analysis.py
```

This shows which gestures are confused. Solutions:
- Collect more data for problematic gestures
- Make gestures more visually distinct
- Ensure consistent hand shape during collection

### Gesture Not Responding

- Check cooldown timer (visible on screen)
- Remove hand between repeated gestures
- Ensure good lighting and hand is in frame
- Verify gesture matches training data

## Performance Analysis

Generate detailed accuracy metrics:

```bash
python confusion_matrix_analysis.py
```

Output includes:
- Confusion matrix visualization
- Per-gesture accuracy
- Precision/Recall scores
- Recommended improvements

## System Requirements

- **OS**: Windows 10/11, macOS, Linux
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum
- **Camera**: 720p webcam recommended
- **CPU**: Any modern CPU (no GPU required)

## Dependencies

Core libraries:
- `opencv-python` - Camera capture and display
- `mediapipe` - Hand tracking and landmark detection
- `scikit-learn` - Machine learning classifier
- `numpy` - Numerical operations
- `joblib` - Model serialization
- `python-vlc` - VLC media player control

## Advanced Configuration

### Modify Gesture Cooldown

In `realtime_vlc_FIXED.py`, adjust cooldown times:

```python
gesture_cooldown = 12  # Frames to wait (lower = more responsive)
```

### Change Video Folder

In `realtime_vlc_FIXED.py`, modify:

```python
VIDEOS_FOLDER = 'demo_videos'  # Change to your folder name
```

### Add More Videos

Simply add more files to `demo_videos/` and update the playlist:

```python
playlist = [
    os.path.join(VIDEOS_FOLDER, 'video1.mp4'),
    os.path.join(VIDEOS_FOLDER, 'video2.mp4'),
    os.path.join(VIDEOS_FOLDER, 'video3.mp4'),
    os.path.join(VIDEOS_FOLDER, 'video4.mp4'),  # Add more
]
```

## Clean Start

To start fresh with new training data:

```bash
python clean_data.py
python collect_gestures.py
python train_model.py
```

## Diagnostics

Check system status:

```bash
python diagnose.py
```

Verifies:
- Python packages installed
- Camera accessible
- VLC available
- Training data present
- Model file exists

## License

MIT License - Free to use and modify

## Acknowledgments

- **MediaPipe** by Google for hand tracking
- **scikit-learn** for machine learning tools
- **OpenCV** for computer vision
- **VLC** for media player control

## Contributing

Contributions welcome! Areas for improvement:
- Additional gesture support
- Multi-hand tracking
- Gesture customization UI
- Mobile app version
- Different media player integrations

## Support

For issues or questions:
1. Check Troubleshooting section above
2. Run diagnostics: `python diagnose.py`
3. Review confusion matrix for accuracy issues
4. Open an issue with diagnostic output

---

**Note**: This system is designed for personal use and experimentation. For production applications, consider additional error handling, security measures, and user accessibility features.
