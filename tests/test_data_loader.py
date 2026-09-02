"""
Unit tests for data loader and class imbalance handler.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
import pandas as pd
from src.data_loader import load_dataset, generate_synthetic_cic_ids
from src.imbalance_handler import get_class_weights, apply_smote_resampling


def test_synthetic_data_generation():
    n_samples = 1000
    df = generate_synthetic_cic_ids(n_samples=n_samples)
    assert len(df) == n_samples
    assert "Label" in df.columns
    assert "Flow Duration" in df.columns
    assert "Flow Bytes/s" in df.columns
    assert df["Label"].nunique() >= 4


def test_class_weights_computation():
    y = np.array(["BENIGN"] * 80 + ["DoS"] * 15 + ["Bot"] * 5)
    weights = get_class_weights(y)
    assert "BENIGN" in weights
    assert "DoS" in weights
    assert "Bot" in weights
    # Minority class "Bot" should receive the highest weight
    assert weights["Bot"] > weights["DoS"] > weights["BENIGN"]


def test_smote_resampling():
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = np.array(["BENIGN"] * 160 + ["Attack"] * 40)
    X_res, y_res = apply_smote_resampling(X, y)
    assert len(y_res) > len(y)
    counts = pd.Series(y_res).value_counts()
    assert counts["BENIGN"] == counts["Attack"]
