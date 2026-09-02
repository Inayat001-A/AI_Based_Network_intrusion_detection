"""
Master Model Training & Benchmarking Pipeline for AI-Based NIDS
Trains and benchmarks Baseline, Ensemble, and Deep Learning models.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.neural_models import NIDS_CNN1D, NIDS_LSTM, train_pytorch_model
from src.hyperparameter_tuner import tune_xgboost, tune_lightgbm

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def load_processed_data():
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Processed datasets not found in data/processed/. Run preprocessor.py first.")
        
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    X_train = df_train.drop(columns=["Target"]).values
    y_train = df_train["Target"].values
    X_test = df_test.drop(columns=["Target"]).values
    y_test = df_test["Target"].values
    
    return X_train, y_train, X_test, y_test


def evaluate_model(model_name, clf, X_train, y_train, X_test, y_test, is_pytorch=False):
    print(f"\n[*] Training & Benchmarking: {model_name}...")
    start_train = time.time()
    
    if not is_pytorch:
        clf.fit(X_train, y_train)
        train_time = time.time() - start_train
        
        # Inference speed benchmark (1,000 samples)
        sample_size = min(1000, len(X_test))
        start_inf = time.time()
        preds = clf.predict(X_test)
        inf_time_ms = ((time.time() - start_inf) / len(X_test)) * 1000 * 1000 # ms per 1k flows
    else:
        # PyTorch training
        trained_model, _ = train_pytorch_model(clf, X_train, y_train, X_test, y_test, epochs=12, batch_size=128)
        train_time = time.time() - start_train
        
        import torch
        trained_model.eval()
        with torch.no_grad():
            start_inf = time.time()
            tensor_test = torch.tensor(X_test, dtype=torch.float32)
            outputs = trained_model(tensor_test)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            inf_time_ms = ((time.time() - start_inf) / len(X_test)) * 1000 * 1000
        clf = trained_model

    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, average="macro", zero_division=0))
    rec = float(recall_score(y_test, preds, average="macro", zero_division=0))
    f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
    
    metrics = {
        "model": model_name,
        "accuracy": round(acc, 4),
        "precision_macro": round(prec, 4),
        "recall_macro": round(rec, 4),
        "f1_macro": round(f1, 4),
        "train_time_sec": round(train_time, 2),
        "inference_latency_ms_per_1k": round(inf_time_ms, 2)
    }
    
    print(f"  • Accuracy:        {acc*100:.2f}%")
    print(f"  • Macro F1-Score:  {f1:.4f}")
    print(f"  • Macro Recall:    {rec:.4f}")
    print(f"  • Latency/1k flows:{inf_time_ms:.2f} ms")
    return metrics, clf


def run_training_pipeline(model_choice="all", tune_params=False):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, y_train, X_test, y_test = load_processed_data()
    num_classes = len(np.unique(y_train))
    input_dim = X_train.shape[1]
    
    print("=" * 70)
    print(f"       AI-BASED NIDS: MODEL TRAINING & BENCHMARKING")
    print(f"       Train Samples: {len(X_train):,} | Test Samples: {len(X_test):,}")
    print("=" * 70)
    
    benchmarks = []
    models_dict = {}
    
    # 1. Baseline: Logistic Regression
    if model_choice in ["all", "lr", "logistic"]:
        clf_lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        m_lr, trained_lr = evaluate_model("Logistic Regression (Baseline)", clf_lr, X_train, y_train, X_test, y_test)
        benchmarks.append(m_lr)
        models_dict["Logistic Regression"] = trained_lr

    # 2. Baseline: Random Forest
    if model_choice in ["all", "rf", "random_forest"]:
        clf_rf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)
        m_rf, trained_rf = evaluate_model("Random Forest (Tree Baseline)", clf_rf, X_train, y_train, X_test, y_test)
        benchmarks.append(m_rf)
        models_dict["Random Forest"] = trained_rf

    # 3. XGBoost
    if model_choice in ["all", "xgb", "xgboost"]:
        xgb_params = {"eval_metric": "mlogloss", "random_state": 42, "n_jobs": -1}
        if tune_params:
            best_p = tune_xgboost(pd.DataFrame(X_train), pd.Series(y_train))
            xgb_params.update(best_p)
        else:
            xgb_params.update({"n_estimators": 120, "max_depth": 6, "learning_rate": 0.1})
            
        clf_xgb = XGBClassifier(**xgb_params)
        m_xgb, trained_xgb = evaluate_model("XGBoost Classifier", clf_xgb, X_train, y_train, X_test, y_test)
        benchmarks.append(m_xgb)
        models_dict["XGBoost"] = trained_xgb

    # 4. LightGBM
    if model_choice in ["all", "lgbm", "lightgbm"]:
        lgb_params = {"random_state": 42, "verbose": -1, "n_jobs": -1}
        if tune_params:
            best_p = tune_lightgbm(pd.DataFrame(X_train), pd.Series(y_train))
            lgb_params.update(best_p)
        else:
            lgb_params.update({"n_estimators": 120, "max_depth": 6, "num_leaves": 31, "learning_rate": 0.1})
            
        clf_lgb = LGBMClassifier(**lgb_params)
        m_lgb, trained_lgb = evaluate_model("LightGBM Classifier", clf_lgb, X_train, y_train, X_test, y_test)
        benchmarks.append(m_lgb)
        models_dict["LightGBM"] = trained_lgb

    # 5. Deep 1D-CNN
    if model_choice in ["all", "cnn", "cnn1d"]:
        cnn_model = NIDS_CNN1D(input_dim=input_dim, num_classes=num_classes)
        m_cnn, trained_cnn = evaluate_model("1D-CNN Deep Neural Net", cnn_model, X_train, y_train, X_test, y_test, is_pytorch=True)
        benchmarks.append(m_cnn)

    # 6. Deep LSTM
    if model_choice in ["all", "lstm"]:
        lstm_model = NIDS_LSTM(input_dim=input_dim, hidden_dim=64, num_classes=num_classes)
        m_lstm, trained_lstm = evaluate_model("LSTM Sequence Neural Net", lstm_model, X_train, y_train, X_test, y_test, is_pytorch=True)
        benchmarks.append(m_lstm)

    # Export Benchmarks Summary
    benchmarks_path = MODELS_DIR / "model_benchmarks.json"
    with open(benchmarks_path, "w") as f:
        json.dump(benchmarks, f, indent=4)
    print(f"\n[+] Saved model benchmark results to: {benchmarks_path}")
    
    # Save Best Tabular Model
    tabular_benchmarks = [b for b in benchmarks if b["model"] in ["XGBoost Classifier", "LightGBM Classifier", "Random Forest (Tree Baseline)", "Logistic Regression (Baseline)"]]
    if tabular_benchmarks:
        best_tabular_meta = max(tabular_benchmarks, key=lambda x: x["f1_macro"])
        best_name = best_tabular_meta["model"].split()[0]
        best_model_obj = models_dict.get(best_name, list(models_dict.values())[0])
        
        best_model_path = MODELS_DIR / "nids_best_model.joblib"
        joblib.dump(best_model_obj, best_model_path)
        print(f"[+] Serialized Best Production Model ({best_tabular_meta['model']}) to: {best_model_path}")

    print("\n--- FINAL BENCHMARK SUMMARY ---")
    summary_df = pd.DataFrame(benchmarks)[["model", "accuracy", "f1_macro", "recall_macro", "inference_latency_ms_per_1k", "train_time_sec"]]
    print(summary_df.to_string(index=False))
    print("=" * 70)
    return benchmarks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NIDS Models")
    parser.add_argument("--model", type=str, default="all", choices=["all", "xgb", "lgbm", "rf", "lr", "cnn", "lstm"])
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    args = parser.parse_args()
    
    run_training_pipeline(model_choice=args.model, tune_params=args.tune)
