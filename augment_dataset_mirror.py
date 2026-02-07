"""
Dataset Mirroring/Augmentation Tool
====================================

This script creates mirrored versions of your existing dataset by flipping
hand landmarks horizontally. This effectively doubles your dataset size
and adds left/right hand variation.

Usage:
    python augment_dataset_mirror.py
    
This will:
1. Load your existing hand_gestures_data.npz
2. Create mirrored versions of all samples
3. Save to hand_gestures_data_augmented.npz (original + mirrored)
"""

import numpy as np
import os

def mirror_landmarks(features):
    """
    Mirror/flip hand landmarks by negating x-coordinates.
    
    Args:
        features: 63-dim flattened array (21 landmarks * 3 coords)
    
    Returns:
        Mirrored 63-dim array
    """
    # Reshape to (21, 3) for easier manipulation
    landmarks = features.reshape(21, 3)
    
    # Flip x-coordinates (negate them after normalization)
    # Since landmarks are normalized (wrist at origin), we negate x
    landmarks[:, 0] = -landmarks[:, 0]
    
    return landmarks.flatten()

def main():
    input_file = "hand_gestures_data.npz"
    output_file = "hand_gestures_data_augmented.npz"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        print("Please run collect_gestures.py first to create a dataset.")
        return
    
    print("=" * 60)
    print("DATASET MIRRORING/AUGMENTATION")
    print("=" * 60)
    
    # Load existing dataset
    print(f"\nLoading {input_file}...")
    data = np.load(input_file, allow_pickle=True)
    X_original = data["X"]
    y_original = data["y"]
    
    # Handle different numpy save formats
    if X_original.ndim == 0:
        X_original = X_original.item()
    if y_original.ndim == 0:
        y_original = y_original.item()
    
    print(f"Original dataset: {len(X_original)} samples")
    
    # Create mirrored versions
    print("\nCreating mirrored versions...")
    X_mirrored = []
    y_mirrored = []
    
    for i, (features, label) in enumerate(zip(X_original, y_original)):
        mirrored_features = mirror_landmarks(features)
        X_mirrored.append(mirrored_features)
        y_mirrored.append(label)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(X_original)} samples...")
    
    # Combine original + mirrored
    X_combined = np.vstack([X_original, np.array(X_mirrored)])
    y_combined = np.hstack([y_original, np.array(y_mirrored)])
    
    print(f"\nAugmented dataset: {len(X_combined)} samples (doubled!)")
    
    # Save augmented dataset
    print(f"\nSaving to {output_file}...")
    np.savez(
        output_file,
        X=X_combined.astype(np.float32),
        y=y_combined.astype(np.int64),
        gesture_names=data.get("gesture_names", np.array([])),
        timestamp=data.get("timestamp", ""),
        augmented=True
    )
    
    print("✓ Augmentation complete!")
    print(f"\nOriginal: {len(X_original)} samples")
    print(f"Mirrored: {len(X_mirrored)} samples")
    print(f"Total:    {len(X_combined)} samples")
    print(f"\nSaved to: {output_file}")
    print("\nTo use the augmented dataset:")
    print("  1. Rename hand_gestures_data.npz to hand_gestures_data_backup.npz")
    print("  2. Rename hand_gestures_data_augmented.npz to hand_gestures_data.npz")
    print("  3. Run train_model.py")

if __name__ == "__main__":
    main()
