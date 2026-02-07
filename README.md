# Real-Time Hand Gesture Recognition System

A high-performance hand gesture recognition system using webcam input, classifying 8 static gestures with >95% accuracy and <50ms latency on laptop CPU.

## Features

- **8 Static Gestures**: Customizable gesture set
- **High Accuracy**: >95% target accuracy
- **Low Latency**: <50ms per frame on CPU
- **Real-time Processing**: Smooth webcam-based recognition
- **Comprehensive Analysis**: Confusion matrix and detailed metrics

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect Training Data

```bash
python collect_gestures.py
```

**Instructions**:
- Press keys `0-7` to label current gesture
- Collect 300-500 samples per gesture
- Press `s` to save progress
- Press `q` to quit and save

### 3. Train Model

```bash
python train_model.py
```

This will:
- Train multiple classifiers (SVM, Random Forest, k-NN)
- Select the best performing one
- Save model to `gesture_classifier.joblib`

### 4. Analyze Performance

```bash
python confusion_matrix_analysis.py
```

This will:
- Generate confusion matrix visualization
- Identify confused gesture pairs
- Provide per-gesture accuracy metrics
- Give recommendations for improvement

### 5. Run Real-Time Recognition

```bash
python realtime_recognition.py
```

## Project Structure

```
.
├── collect_gestures.py          # Data collection script
├── train_model.py               # Model training script
├── confusion_matrix_analysis.py # Performance analysis tool
├── realtime_recognition.py      # Real-time recognition (to be created)
├── requirements.txt             # Python dependencies
├── DATA_COLLECTION_GUIDE.md     # Detailed collection guide
└── README.md                    # This file
```

## Gesture Definitions

Default 8 gestures (customizable in scripts):
- `0`: PALM
- `1`: FIST
- `2`: THUMBS_UP
- `3`: THUMBS_DOWN
- `4`: POINT
- `5`: V_SIGN
- `6`: OK
- `7`: ROCK

## Data Collection Best Practices

See `DATA_COLLECTION_GUIDE.md` for comprehensive guidance. Key points:

1. **Sample Size**: 300-500 samples per gesture
2. **Variation**: Different distances, rotations, lighting
3. **Distinctness**: Ensure gestures are visually different
4. **Balance**: Roughly equal samples per gesture

## Understanding Confusion Analysis

The `confusion_matrix_analysis.py` script helps you:

1. **Identify Confused Pairs**: See which gestures are mistaken for each other
2. **Per-Gesture Accuracy**: Check individual gesture performance
3. **Precision/Recall**: Understand false positives and negatives
4. **Get Recommendations**: Actionable steps to improve accuracy

### Reading the Confusion Matrix

- **Diagonal values**: Correct predictions (higher is better)
- **Off-diagonal values**: Confusions (lower is better)
- **Percentages**: Show what percentage of each gesture is misclassified

### Common Issues

- **High confusion between two gestures**: Make them more distinct or collect more samples
- **Low accuracy for one gesture**: Collect more diverse samples for that gesture
- **Class imbalance**: Collect more samples for underrepresented gestures

## Performance Optimization

For <50ms latency:

1. **Lower resolution**: 640x480 (default)
2. **Single hand tracking**: Only track one hand
3. **Efficient classifier**: SVM or Random Forest (not deep learning)
4. **Temporal smoothing**: Majority vote over last 5-7 frames

## Troubleshooting

### Low Accuracy (<95%)

1. Run `confusion_matrix_analysis.py` to identify issues
2. Collect more data for problematic gestures
3. Ensure gestures are visually distinct
4. Check class balance

### High Latency (>50ms)

1. Reduce frame resolution
2. Use simpler classifier (Linear SVM)
3. Reduce smoothing window size
4. Check CPU performance

### Hand Not Detected

1. Ensure good lighting
2. Use plain background
3. Keep hand in frame
4. Check webcam permissions

## License

MIT License - feel free to use and modify for your projects.
