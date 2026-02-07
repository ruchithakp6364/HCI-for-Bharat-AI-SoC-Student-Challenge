"""
Training Script with Detailed Metrics
=====================================

Trains a gesture classifier and provides comprehensive evaluation metrics.
"""

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

GESTURE_NAMES = {
    0: "PALM",
    1: "FIST", 
    2: "THUMBS_UP",
    3: "THUMBS_DOWN",
    4: "POINT",
    5: "V_SIGN",
    6: "OK",
    7: "ROCK",
}

def load_dataset(filename="hand_gestures_data.npz"):
    """Load dataset."""
    if not os.path.exists(filename):
        print(f"Error: Dataset file '{filename}' not found!")
        return None, None
    
    data = np.load(filename, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    
    if X.ndim == 0:
        X = X.item()
    if y.ndim == 0:
        y = y.item()
    
    return X, y

def main():
    print("=" * 60)
    print("TRAINING GESTURE CLASSIFIER")
    print("=" * 60)
    
    # Load data
    X, y = load_dataset()
    if X is None:
        return
    
    print(f"\nLoaded {len(X)} samples")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Try different classifiers
    classifiers = {
        "SVM (RBF)": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10.0, gamma="scale", probability=False)
        ),
        "SVM (Linear)": make_pipeline(
            StandardScaler(),
            SVC(kernel="linear", C=1.0, probability=False)
        ),
        "Random Forest": make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        ),
        "k-NN (k=5)": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        ),
    }
    
    best_clf = None
    best_score = 0
    best_name = None
    
    print("\n" + "-" * 60)
    print("EVALUATING CLASSIFIERS")
    print("-" * 60)
    
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train, y_train)
        
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"  Train Accuracy: {train_score * 100:.2f}%")
        print(f"  Test Accuracy:  {test_score * 100:.2f}%")
        print(f"  CV Accuracy:    {cv_mean * 100:.2f}% (+/- {cv_std * 100:.2f}%)")
        
        if test_score > best_score:
            best_score = test_score
            best_clf = clf
            best_name = name
    
    print("\n" + "=" * 60)
    print(f"BEST CLASSIFIER: {best_name}")
    print(f"Test Accuracy: {best_score * 100:.2f}%")
    print("=" * 60)
    
    # Save best model
    joblib.dump(best_clf, "gesture_classifier.joblib")
    print(f"\n✓ Saved best classifier to gesture_classifier.joblib")
    
    if best_score >= 0.95:
        print("\n✓ Target accuracy (>95%) achieved!")
    else:
        print(f"\n⚠ Accuracy below target. Current: {best_score * 100:.2f}%, Target: 95%")
        print("  Run confusion_matrix_analysis.py to identify issues.")

if __name__ == "__main__":
    main()
