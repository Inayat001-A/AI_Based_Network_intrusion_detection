"""
Unit tests for AI & ML models, Neural Networks, and Autoencoder anomaly detector.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.neural_models import NIDS_CNN1D, NIDS_LSTM
from src.autoencoder import ZeroDayAutoencoderDetector


def test_neural_models_forward():
    batch_size = 16
    input_dim = 30
    num_classes = 5
    dummy_input = torch.randn(batch_size, input_dim)
    
    # Test 1D-CNN
    cnn = NIDS_CNN1D(input_dim=input_dim, num_classes=num_classes)
    cnn_out = cnn(dummy_input)
    assert cnn_out.shape == (batch_size, num_classes)
    
    # Test LSTM
    lstm = NIDS_LSTM(input_dim=input_dim, hidden_dim=32, num_classes=num_classes)
    lstm_out = lstm(dummy_input)
    assert lstm_out.shape == (batch_size, num_classes)


def test_autoencoder_anomaly_detection(tmp_path):
    np.random.seed(42)
    # Generate synthetic benign traffic (normal distribution near 0)
    X_benign = np.random.normal(loc=0.0, scale=0.5, size=(500, 30))
    # Generate synthetic attack anomalies (shifted distribution)
    X_attack = np.random.normal(loc=5.0, scale=2.0, size=(100, 30))
    
    detector = ZeroDayAutoencoderDetector(input_dim=30, latent_dim=4, threshold_percentile=95.0)
    detector.fit(X_benign, epochs=5, batch_size=64)
    
    assert detector.threshold is not None
    assert detector.threshold > 0
    
    # Predict anomalies
    is_anomaly_benign, errors_benign = detector.predict_anomalies(X_benign[:50])
    is_anomaly_attack, errors_attack = detector.predict_anomalies(X_attack)
    
    # Attack traffic reconstruction error should be significantly higher than benign
    assert np.mean(errors_attack) > np.mean(errors_benign)
    # Attacks should have high detection rate
    assert np.sum(is_anomaly_attack) > len(X_attack) * 0.70
    
    # Test serialization & reloading
    detector.save(models_dir=tmp_path)
    loaded_detector = ZeroDayAutoencoderDetector.load(models_dir=tmp_path)
    assert loaded_detector.threshold == detector.threshold
    
    reloaded_is_anom, _ = loaded_detector.predict_anomalies(X_attack[:10])
    np.testing.assert_array_equal(is_anomaly_attack[:10], reloaded_is_anom)


def test_tree_ensembles_training():
    np.random.seed(42)
    X = np.random.randn(200, 30)
    y = np.random.choice([0, 1, 2], size=200)
    
    # XGBoost
    xgb = XGBClassifier(n_estimators=10, max_depth=3, eval_metric="mlogloss", random_state=42)
    xgb.fit(X, y)
    xgb_preds = xgb.predict(X[:10])
    assert len(xgb_preds) == 10
    
    # LightGBM
    lgb = LGBMClassifier(n_estimators=10, max_depth=3, random_state=42, verbose=-1)
    lgb.fit(X, y)
    lgb_preds = lgb.predict(X[:10])
    assert len(lgb_preds) == 10
