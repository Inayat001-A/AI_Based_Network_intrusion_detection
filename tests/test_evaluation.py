"""
Unit tests for Comprehensive Evaluation, FPR, and metric reporting.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
import pandas as pd
from src.evaluate import calculate_false_positive_rate, run_evaluation, EVAL_FIGURES_DIR, REPORTS_DIR


def test_calculate_false_positive_rate():
    # 0 = Benign, 1 = Attack
    # 100 samples: 80 benign (0), 20 attacks (1)
    y_true = np.array([0] * 80 + [1] * 20)
    
    # Predict 78 benign correctly, 2 benign falsely as attack (FP=2)
    # Predict 19 attacks correctly, 1 attack missed as benign (FN=1)
    y_pred = np.array([0] * 78 + [1] * 2 + [0] * 1 + [1] * 19)
    
    fpr, fp, tn, tp, fn = calculate_false_positive_rate(y_true, y_pred, benign_class_idx=0)
    
    assert fp == 2
    assert tn == 78
    assert tp == 19
    assert fn == 1
    assert pytest.approx(fpr, 0.001) == 2 / 80 # 2.5% FPR


def test_full_evaluation_pipeline():
    eval_report = run_evaluation()
    
    assert eval_report is not None
    assert eval_report["overall_accuracy"] > 0.95
    assert eval_report["macro_f1_score"] > 0.95
    assert eval_report["false_positive_rate"] < 0.01  # Less than 1% false alarm rate
    
    # Check generated figures
    assert (EVAL_FIGURES_DIR / "01_confusion_matrix.png").exists()
    assert (EVAL_FIGURES_DIR / "02_roc_curves.png").exists()
    assert (EVAL_FIGURES_DIR / "03_precision_recall_curves.png").exists()
    assert (EVAL_FIGURES_DIR / "04_per_class_metrics.png").exists()
    assert (REPORTS_DIR / "evaluation_report.json").exists()
