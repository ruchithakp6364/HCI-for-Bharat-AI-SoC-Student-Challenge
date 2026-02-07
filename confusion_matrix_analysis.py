"""
Confusion Matrix Analysis Tool
==============================

This script helps you identify:
1. Which gesture pairs are confused by the model
2. Which gestures have low precision/recall
3. Overall model performance breakdown
4. Recommendations for improving accuracy

Usage:
    python confusion_matrix_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import joblib
import os

# Gesture names - must match collect_gestures.py exactly
GESTURE_NAMES = {
    0: "PALM",
    1: "THUMBS_RIGHT",
    2: "THUMBS_LEFT",
    3: "TWO_FINGERS_UP",
    4: "TWO_FINGERS_DOWN",
    5: "OK",
}

# Derived so analysis adapts to your gesture set
VALID_GESTURE_INDICES = sorted(GESTURE_NAMES.keys())

def normalize_landmarks(landmarks):
    """Same normalization as in collection script."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    wrist = pts[0].copy()
    pts -= wrist
    scale = np.linalg.norm(pts[9])
    if scale < 1e-6:
        scale = 1.0
    pts /= scale
    return pts.flatten()

def load_dataset(filename="hand_gestures_data.npz"):
    """Load dataset."""
    if not os.path.exists(filename):
        print(f"Error: Dataset file '{filename}' not found!")
        print("Please run collect_gestures.py first to collect data.")
        return None, None
    
    data = np.load(filename, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    
    # Handle different numpy save formats
    if X.ndim == 0:
        X = X.item()
    if y.ndim == 0:
        y = y.item()
    
    return X, y

def train_and_evaluate(X, y, test_size=0.2, random_state=42):
    """Train classifier and return predictions for analysis."""
    print("=" * 60)
    print("TRAINING CLASSIFIER")
    print("=" * 60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train classifier
    print("\nTraining SVM classifier...")
    clf = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10.0, gamma="scale", probability=False)
    )
    
    clf.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    # Calculate accuracies
    train_acc = np.mean(y_train_pred == y_train)
    test_acc = np.mean(y_test_pred == y_test)
    
    print(f"\nTraining Accuracy: {train_acc * 100:.2f}%")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    
    # Save model
    joblib.dump(clf, "gesture_classifier.joblib")
    print("\n✓ Saved classifier to gesture_classifier.joblib")
    
    return clf, y_test, y_test_pred, y_train, y_train_pred

def analyze_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    """Create and analyze confusion matrix."""
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX ANALYSIS")
    print("=" * 60)
    
    # Create confusion matrix (labels = your gesture indices)
    cm = confusion_matrix(y_true, y_pred, labels=VALID_GESTURE_INDICES)
    
    # Calculate percentages (normalized by true labels)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    cm_percent = np.nan_to_num(cm_percent)  # Handle division by zero
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Absolute counts
    gesture_labels = [GESTURE_NAMES[i] for i in VALID_GESTURE_INDICES]
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=gesture_labels,
        yticklabels=gesture_labels,
        ax=axes[0],
        cbar_kws={'label': 'Count'}
    )
    axes[0].set_title('Confusion Matrix (Absolute Counts)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted Gesture', fontsize=12)
    axes[0].set_ylabel('True Gesture', fontsize=12)
    axes[0].tick_params(axis='both', labelsize=9)
    
    # Plot 2: Percentages
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt='.1f',
        cmap='Oranges',
        xticklabels=gesture_labels,
        yticklabels=gesture_labels,
        ax=axes[1],
        cbar_kws={'label': 'Percentage (%)'}
    )
    axes[1].set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted Gesture', fontsize=12)
    axes[1].set_ylabel('True Gesture', fontsize=12)
    axes[1].tick_params(axis='both', labelsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved confusion matrix to {save_path}")
    
    # Analyze confusion patterns
    print("\n" + "-" * 60)
    print("CONFUSION ANALYSIS")
    print("-" * 60)
    
    # Find most confused pairs
    confused_pairs = []
    n = len(VALID_GESTURE_INDICES)
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                percentage = cm_percent[i, j]
                confused_pairs.append((VALID_GESTURE_INDICES[i], VALID_GESTURE_INDICES[j], cm[i, j], percentage))
    
    # Sort by confusion count
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    
    if confused_pairs:
        print("\nTop Confused Gesture Pairs:")
        print(f"{'True Gesture':<20} {'Predicted As':<20} {'Count':<10} {'Percentage':<10}")
        print("-" * 60)
        for true_idx, pred_idx, count, pct in confused_pairs[:10]:
            print(f"{GESTURE_NAMES[true_idx]:<20} {GESTURE_NAMES[pred_idx]:<20} {count:<10} {pct:>6.2f}%")
    else:
        print("\n✓ No confusions detected!")
    
    # Per-gesture accuracy
    print("\n" + "-" * 60)
    print("PER-GESTURE ACCURACY")
    print("-" * 60)
    print(f"{'Gesture':<20} {'Correct':<10} {'Total':<10} {'Accuracy':<10} {'Status':<15}")
    print("-" * 60)
    
    for i in range(len(VALID_GESTURE_INDICES)):
        idx = VALID_GESTURE_INDICES[i]
        correct = cm[i, i]
        total = cm[i, :].sum()
        accuracy = (correct / total * 100) if total > 0 else 0
        status = "✓ EXCELLENT" if accuracy >= 95 else \
                 "⚠ GOOD" if accuracy >= 90 else \
                 "✗ NEEDS WORK"
        print(f"{GESTURE_NAMES[idx]:<20} {correct:<10} {total:<10} {accuracy:>6.2f}% {status:<15}")
    
    return cm, cm_percent, confused_pairs

def detailed_classification_report(y_true, y_pred):
    """Print detailed classification metrics."""
    print("\n" + "=" * 60)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 60)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=VALID_GESTURE_INDICES, zero_division=0
    )
    
    print(f"\n{'Gesture':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    
    for i in range(len(VALID_GESTURE_INDICES)):
        idx = VALID_GESTURE_INDICES[i]
        print(f"{GESTURE_NAMES[idx]:<20} {precision[i]:>10.2%} {recall[i]:>10.2%} "
              f"{f1[i]:>10.2%} {int(support[i]):>10}")
    
    # Overall metrics
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)
    
    print("-" * 70)
    print(f"{'Macro Avg':<20} {macro_precision:>10.2%} {macro_recall:>10.2%} "
          f"{macro_f1:>10.2%} {int(support.sum()):>10}")
    print(f"{'Weighted Avg':<20} {weighted_precision:>10.2%} {weighted_recall:>10.2%} "
          f"{weighted_f1:>10.2%} {int(support.sum()):>10}")
    
    return precision, recall, f1, support

def generate_recommendations(cm, cm_percent, precision, recall, confused_pairs):
    """Generate actionable recommendations based on analysis."""
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS FOR IMPROVING ACCURACY")
    print("=" * 60)
    
    recommendations = []
    
    # Check for low-performing gestures
    for i in range(len(VALID_GESTURE_INDICES)):
        accuracy = cm[i, i] / cm[i, :].sum() * 100 if cm[i, :].sum() > 0 else 0
        if accuracy < 95:
            recommendations.append(
                f"⚠ {GESTURE_NAMES[VALID_GESTURE_INDICES[i]]} has {accuracy:.1f}% accuracy. "
                f"Consider collecting more diverse samples."
            )
    
    # Check for confused pairs
    if confused_pairs:
        top_confusion = confused_pairs[0]
        true_idx, pred_idx, count, pct = top_confusion
        if pct > 5:  # More than 5% confusion
            recommendations.append(
                f"⚠ {GESTURE_NAMES[true_idx]} is confused with {GESTURE_NAMES[pred_idx]} "
                f"({pct:.1f}% of cases). These gestures may be too similar. "
                f"Consider: (1) Making gestures more distinct, "
                f"(2) Collecting more samples for both, "
                f"(3) Adding features that distinguish them."
            )
    
    # Check for low precision (false positives)
    for i in range(len(VALID_GESTURE_INDICES)):
        if precision[i] < 0.90:
            false_positives = cm[:, i].sum() - cm[i, i]
            recommendations.append(
                f"⚠ {GESTURE_NAMES[VALID_GESTURE_INDICES[i]]} has low precision ({precision[i]:.1%}). "
                f"{false_positives} samples are incorrectly predicted as this gesture. "
                f"Other gestures may be too similar to this one."
            )
    
    # Check for low recall (false negatives)
    for i in range(len(VALID_GESTURE_INDICES)):
        if recall[i] < 0.90:
            false_negatives = cm[i, :].sum() - cm[i, i]
            recommendations.append(
                f"⚠ {GESTURE_NAMES[VALID_GESTURE_INDICES[i]]} has low recall ({recall[i]:.1%}). "
                f"{false_negatives} samples of this gesture are misclassified. "
                f"Collect more diverse samples for this gesture."
            )
    
    # Check for class imbalance
    class_counts = cm.sum(axis=1)
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    if imbalance_ratio > 2.0:
        recommendations.append(
            f"⚠ Class imbalance detected ({imbalance_ratio:.1f}x ratio). "
            f"Collect more samples for underrepresented gestures."
        )
    
    if recommendations:
        print("\nIssues Found:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("\n✓ No major issues detected! Model performance looks good.")
    
    # General best practices
    print("\n" + "-" * 60)
    print("GENERAL BEST PRACTICES:")
    print("-" * 60)
    print("1. Collect 400+ samples per gesture for robust models")
    print("2. Ensure gestures are visually distinct (different finger configurations)")
    print("3. Collect data with variation:")
    print("   - Different distances from camera")
    print("   - Different hand rotations")
    print("   - Both left and right hands")
    print("   - Multiple lighting conditions")
    print("4. If two gestures are frequently confused:")
    print("   - Make them more visually distinct")
    print("   - Collect more samples specifically for those pairs")
    print("   - Consider combining them into one gesture if they're too similar")
    print("5. Test model on data from different sessions/users for generalization")

def main():
    # Load dataset
    print("Loading dataset...")
    X, y = load_dataset()
    if X is None:
        return
    
    print(f"Loaded {len(X)} samples")
    print(f"Features per sample: {X.shape[1]}")
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for idx, count in zip(unique, counts):
        print(f"  {GESTURE_NAMES[idx]}: {count} samples")
    
    # Train and evaluate
    clf, y_test, y_test_pred, y_train, y_train_pred = train_and_evaluate(X, y)
    
    # Analyze confusion matrix
    cm, cm_percent, confused_pairs = analyze_confusion_matrix(y_test, y_test_pred)
    
    # Detailed metrics
    precision, recall, f1, support = detailed_classification_report(y_test, y_test_pred)
    
    # Generate recommendations
    generate_recommendations(cm, cm_percent, precision, recall, confused_pairs)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review confusion_matrix.png to see visual confusion patterns")
    print("2. Follow recommendations above to improve data collection")
    print("3. Re-run this script after collecting more data")
    print("4. Use gesture_classifier.joblib for real-time recognition")

if __name__ == "__main__":
    main()
