"""
Class Imbalance Mitigation Module for AI-Based NIDS
Implements Cost-Sensitive Weighting and Synthetic Minority Over-sampling (SMOTE).
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def get_class_weights(y, return_dict=True):
    """
    Computes balanced class weights inversely proportional to class frequencies.
    Weight = n_samples / (n_classes * np.bincount(y))
    """
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    if return_dict:
        return {cls: weight for cls, weight in zip(classes, weights)}
    return weights


def apply_smote_resampling(X, y, strategy="auto", k_neighbors=5, random_state=42):
    """
    Applies SMOTE (Synthetic Minority Over-sampling Technique) to balance minority attack classes.
    """
    print(f"[*] Original dataset distribution:\n{pd.Series(y).value_counts()}")
    smote = SMOTE(sampling_strategy=strategy, k_neighbors=k_neighbors, random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"[+] Resampled dataset distribution:\n{pd.Series(y_res).value_counts()}")
    return X_res, y_res


def apply_smote_tomek(X, y, random_state=42):
    """
    Applies SMOTE + Tomek Links to simultaneously oversample minority attacks
    and remove ambiguous borderline noise points.
    """
    print(f"[*] Applying SMOTE-Tomek combined resampling...")
    smt = SMOTETomek(random_state=random_state)
    X_res, y_res = smt.fit_resample(X, y)
    print(f"[+] SMOTE-Tomek complete. New shape: {X_res.shape}")
    return X_res, y_res


if __name__ == "__main__":
    from src.data_loader import load_dataset
    
    df = load_dataset(sample_size=10000)
    label_col = "Label" if "Label" in df.columns else "label"
    X = df.select_dtypes(include=[np.number]).dropna()
    y = df.loc[X.index, label_col]
    
    print("\n--- Testing Class Weights Computation ---")
    weights = get_class_weights(y)
    for cls, w in weights.items():
        print(f"  Class '{cls}': weight = {w:.4f}")
        
    print("\n--- Testing SMOTE Resampling ---")
    X_res, y_res = apply_smote_resampling(X, y)
    print(f"Original samples: {len(X):,} | Resampled samples: {len(X_res):,}")
