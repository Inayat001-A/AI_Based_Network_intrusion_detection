"""
Comprehensive Evaluation and Security Reporting Module for AI-Based NIDS
Computes Detection Rate, False Alarm Rate, Confusion Matrix, ROC-AUC, and PR Curves.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.preprocessing import label_binarize

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
EVAL_FIGURES_DIR = REPORTS_DIR / "eval_figures"

# Set visualization style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


def setup_eval_dirs():
    EVAL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def calculate_false_positive_rate(y_true, y_pred, benign_class_idx=0):
    """
    Computes False Positive Rate (FPR / False Alarm Rate) on Benign class.
    FPR = FP / (FP + TN) where 'positive' means classified as an attack.
    """
    # Binary conversion: 0 = Benign, 1 = Attack
    y_true_binary = (y_true != benign_class_idx).astype(int)
    y_pred_binary = (y_pred != benign_class_idx).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return fpr, int(fp), int(tn), int(tp), int(fn)


def run_evaluation(model_path=MODELS_DIR / "nids_best_model.joblib", test_csv=PROCESSED_DIR / "test.csv"):
    setup_eval_dirs()
    print("=" * 70)
    print("      AI-BASED NIDS: PHASE 4 COMPREHENSIVE EVALUATION")
    print("=" * 70)
    
    # 1. Load Model & Preprocessor Metadata
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    
    model = joblib.load(model_path)
    features_json_path = MODELS_DIR / "selected_features.json"
    
    if features_json_path.exists():
        with open(features_json_path, "r") as f:
            feat_meta = json.load(f)
        class_names = feat_meta["classes"]
    else:
        class_names = ["BENIGN", "DoS Hulk", "PortScan", "SSH-Patator", "Bot"]
        
    n_classes = len(class_names)
    print(f"[+] Loaded Model: {type(model).__name__}")
    print(f"[+] Threat Classes ({n_classes}): {class_names}")
    
    # 2. Ingest Test Data
    df_test = pd.read_csv(test_csv)
    X_test = df_test.drop(columns=["Target"]).values
    y_test = df_test["Target"].values
    print(f"[+] Ingested Test Data: {len(X_test):,} flows with {X_test.shape[1]} features.")
    
    # 3. Model Inference & Probability Scoring
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time
    throughput_flows_per_sec = len(X_test) / max(inference_time, 0.0001)
    latency_per_flow_us = (inference_time / len(X_test)) * 1_000_000
    
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    else:
        y_score = None
        
    # 4. Compute Metrics
    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    macro_prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    
    fpr, fp_count, tn_count, tp_count, fn_count = calculate_false_positive_rate(y_test, y_pred, benign_class_idx=0)
    
    cls_report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    
    print("\n--- 1. OVERALL CYBERSECURITY METRICS ---")
    print(f"  * Overall Accuracy:                 {acc*100:.2f}%")
    print(f"  * Macro Detection Rate (Recall):    {macro_rec*100:.2f}%")
    print(f"  * Macro Precision:                  {macro_prec*100:.2f}%")
    print(f"  * Macro F1-Score:                   {macro_f1:.4f}")
    print(f"  * False Positive Rate (FPR / FAR):  {fpr*100:.4f}% ({fp_count:,} false alarms out of {tn_count+fp_count:,} benign flows)")
    print(f"  * Inference Throughput:             {throughput_flows_per_sec:,.1f} flows/sec")
    print(f"  * Processing Latency per Flow:      {latency_per_flow_us:.2f} us")
    
    print("\n--- 2. PER-CLASS DETECTION PERFORMANCE ---")
    print(f"{'Threat Class':<25} {'Precision':<12} {'Recall (DR)':<14} {'F1-Score':<10} {'Support':<8}")
    print("-" * 70)
    per_class_summary = {}
    for cls in class_names:
        if cls in cls_report:
            p = cls_report[cls]["precision"]
            r = cls_report[cls]["recall"]
            f = cls_report[cls]["f1-score"]
            s = int(cls_report[cls]["support"])
            per_class_summary[cls] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "support": s}
            print(f"{cls:<25} {p*100:>8.2f}%   {r*100:>10.2f}%   {f:>8.4f}   {s:>7,}")
            
    # 5. Generate Visual Artifacts
    print("\n--- 3. GENERATING HIGH-RESOLUTION REPORTS & FIGURES ---")
    
    # Figure 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2%", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={"label": "Normalized Proportion"}
    )
    plt.title("Normalized Confusion Matrix (Actual vs. Predicted Threats)", fontweight="bold")
    plt.xlabel("Predicted Threat Category")
    plt.ylabel("Actual Threat Category")
    plt.xticks(rotation=25)
    plt.tight_layout()
    fig1_path = EVAL_FIGURES_DIR / "01_confusion_matrix.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {fig1_path}")
    
    # Figure 2: Multi-Class ROC Curves
    roc_auc_scores = {}
    if y_score is not None:
        y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
        plt.figure(figsize=(9, 7))
        
        for i, cls in enumerate(class_names):
            fpr_cls, tpr_cls, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc_val = auc(fpr_cls, tpr_cls)
            roc_auc_scores[cls] = round(float(roc_auc_val), 4)
            plt.plot(fpr_cls, tpr_cls, lw=2, label=f"{cls} (AUC = {roc_auc_val:.4f})")
            
        plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.5000)")
        plt.xlim([-0.02, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (Detection Rate)")
        plt.title("Multi-Class One-vs-Rest ROC Curves", fontweight="bold")
        plt.legend(loc="lower right")
        plt.tight_layout()
        fig2_path = EVAL_FIGURES_DIR / "02_roc_curves.png"
        plt.savefig(fig2_path, dpi=300)
        plt.close()
        print(f"[+] Saved: {fig2_path}")

    # Figure 3: Precision-Recall Curves
    pr_auc_scores = {}
    if y_score is not None:
        plt.figure(figsize=(9, 7))
        for i, cls in enumerate(class_names):
            prec_arr, rec_arr, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
            pr_auc = average_precision_score(y_test_bin[:, i], y_score[:, i])
            pr_auc_scores[cls] = round(float(pr_auc), 4)
            plt.plot(rec_arr, prec_arr, lw=2, label=f"{cls} (AP = {pr_auc:.4f})")
            
        plt.xlabel("Recall (Detection Rate)")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves across Imbalanced Threat Classes", fontweight="bold")
        plt.legend(loc="lower left")
        plt.tight_layout()
        fig3_path = EVAL_FIGURES_DIR / "03_precision_recall_curves.png"
        plt.savefig(fig3_path, dpi=300)
        plt.close()
        print(f"[+] Saved: {fig3_path}")

    # Figure 4: Per-Class Metrics Bar Chart
    df_metrics = pd.DataFrame(per_class_summary).T[["precision", "recall", "f1"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    df_metrics.plot(kind="bar", ax=ax, colormap="viridis", width=0.75)
    plt.title("Per-Class Precision, Detection Rate (Recall) & F1-Score", fontweight="bold")
    plt.xlabel("Threat Class")
    plt.ylabel("Score (0.0 to 1.0)")
    plt.ylim(0.85, 1.02)
    plt.xticks(rotation=20)
    plt.legend(["Precision", "Recall (Detection Rate)", "F1-Score"])
    plt.tight_layout()
    fig4_path = EVAL_FIGURES_DIR / "04_per_class_metrics.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {fig4_path}")

    # 6. Export Comprehensive JSON Summary Report
    eval_report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": type(model).__name__,
        "test_samples": int(len(X_test)),
        "overall_accuracy": round(acc, 4),
        "macro_f1_score": round(macro_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "false_positive_rate": round(fpr, 6),
        "false_alarm_count": fp_count,
        "throughput_flows_per_sec": round(throughput_flows_per_sec, 1),
        "latency_per_flow_microseconds": round(latency_per_flow_us, 2),
        "per_class_metrics": per_class_summary,
        "roc_auc_scores": roc_auc_scores,
        "pr_auc_scores": pr_auc_scores,
        "generated_figures": [
            str(fig1_path.name),
            str(fig2_path.name) if y_score is not None else None,
            str(fig3_path.name) if y_score is not None else None,
            str(fig4_path.name)
        ]
    }
    
    json_path = REPORTS_DIR / "evaluation_report.json"
    with open(json_path, "w") as f:
        json.dump(eval_report, f, indent=4)
    print(f"\n[+] Exported Comprehensive Security Evaluation Report to: {json_path}")
    print("=" * 70)
    return eval_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained NIDS Model")
    parser.add_argument("--model", type=str, default=str(MODELS_DIR / "nids_best_model.joblib"))
    parser.add_argument("--test-data", type=str, default=str(PROCESSED_DIR / "test.csv"))
    args = parser.parse_args()
    
    run_evaluation(model_path=args.model, test_csv=args.test_data)
