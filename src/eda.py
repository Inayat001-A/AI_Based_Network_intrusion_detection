"""
Exploratory Data Analysis (EDA) Module for AI-Based NIDS
Performs statistical profiling, class imbalance analysis, and generates publication-quality charts.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_dataset

# Set style for high-quality visuals
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
FIGURES_DIR = REPORTS_DIR / "eda_figures"


def setup_reports_dir():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def run_eda(dataset_name="cic-ids2017", sample_size=50000):
    setup_reports_dir()
    print("=" * 70)
    print("        AI-BASED NIDS - EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)
    
    # 1. Load Data
    df = load_dataset(dataset_name=dataset_name, sample_size=sample_size)
    print(f"\n[+] Loaded dataset shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    label_col = "Label" if "Label" in df.columns else "label"
    
    # 2. Data Integrity & Missing/Infinite Value Check
    print("\n--- 1. Data Integrity & Missing Values ---")
    numeric_df = df.select_dtypes(include=[np.number])
    
    inf_counts = np.isinf(numeric_df).sum()
    total_infs = int(inf_counts.sum())
    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())
    
    print(f"Total Missing (NaN) Values: {total_nulls:,}")
    print(f"Total Infinite (Inf/-Inf) Values: {total_infs:,}")
    
    # Clean infinities and nulls for analysis
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Usable clean records: {len(df_clean):,} ({len(df_clean)/len(df)*100:.2f}%)")
    
    # 3. Class Distribution & Imbalance
    print("\n--- 2. Class Distribution & Imbalance Ratios ---")
    class_counts = df_clean[label_col].value_counts()
    class_pcts = df_clean[label_col].value_counts(normalize=True) * 100
    
    class_summary = {}
    for cls in class_counts.index:
        cnt = int(class_counts[cls])
        pct = float(class_pcts[cls])
        class_summary[str(cls)] = {"count": cnt, "percentage": round(pct, 2)}
        print(f"  * {cls:<25}: {cnt:>8,} flows ({pct:>5.2f}%)")
        
    majority_class = class_counts.index[0]
    imbalance_ratios = {str(cls): round(float(class_counts[majority_class] / cnt), 2) for cls, cnt in class_counts.items()}
    print(f"\nImbalance Ratio vs. Majority ({majority_class}):")
    for cls, ratio in imbalance_ratios.items():
        print(f"  * {cls:<25}: 1 : {ratio}")
        
    # 4. Generate Visualizations
    print("\n--- 3. Generating High-Resolution Figures ---")
    
    # Figure 1: Class Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    palette = sns.color_palette("tab10", len(class_counts))
    
    sns.barplot(x=class_counts.values, y=class_counts.index, hue=class_counts.index, palette=palette, ax=ax1, legend=False)
    ax1.set_title("Attack Class Frequency (Count)", fontweight="bold")
    ax1.set_xlabel("Number of Flows")
    for i, v in enumerate(class_counts.values):
        ax1.text(v + (max(class_counts.values) * 0.01), i, f"{v:,}", va="center", fontsize=9)
        
    ax2.pie(class_counts.values, labels=class_counts.index, autopct="%1.1f%%", colors=palette, startangle=140, explode=[0.05]*len(class_counts))
    ax2.set_title("Class Composition (%)", fontweight="bold")
    plt.tight_layout()
    fig1_path = FIGURES_DIR / "01_class_distribution.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {fig1_path}")
    
    # Figure 2: Flow Duration by Attack Category
    if "Flow Duration" in df_clean.columns:
        plt.figure(figsize=(10, 5))
        df_clean["Log_Flow_Duration"] = np.log10(np.maximum(df_clean["Flow Duration"], 1.0))
        sns.boxplot(data=df_clean, x=label_col, y="Log_Flow_Duration", palette="Set2")
        plt.title("Log-Transformed Flow Duration by Attack Category", fontweight="bold")
        plt.xlabel("Attack Class")
        plt.ylabel("Log10(Flow Duration in us)")
        plt.xticks(rotation=25)
        plt.tight_layout()
        fig2_path = FIGURES_DIR / "02_flow_duration_by_attack.png"
        plt.savefig(fig2_path, dpi=300)
        plt.close()
        print(f"[+] Saved: {fig2_path}")

    # Figure 3: Packet Length Mean & Variance
    if "Fwd Packet Length Mean" in df_clean.columns and "Bwd Packet Length Mean" in df_clean.columns:
        plt.figure(figsize=(10, 5))
        sns.scatterplot(
            data=df_clean.sample(min(3000, len(df_clean)), random_state=42),
            x="Fwd Packet Length Mean",
            y="Bwd Packet Length Mean",
            hue=label_col,
            alpha=0.7,
            palette="bright"
        )
        plt.title("Forward vs. Backward Packet Length Mean (Bytes)", fontweight="bold")
        plt.xlabel("Fwd Packet Length Mean (Bytes)")
        plt.ylabel("Bwd Packet Length Mean (Bytes)")
        plt.tight_layout()
        fig3_path = FIGURES_DIR / "03_packet_length_scatter.png"
        plt.savefig(fig3_path, dpi=300)
        plt.close()
        print(f"[+] Saved: {fig3_path}")

    # Figure 4: TCP Flags Analysis by Attack Category
    flag_cols = [c for c in ["SYN Flag Count", "ACK Flag Count", "PSH Flag Count", "FIN Flag Count", "RST Flag Count"] if c in df_clean.columns]
    if flag_cols:
        flag_means = df_clean.groupby(label_col)[flag_cols].mean()
        plt.figure(figsize=(11, 5))
        flag_means.plot(kind="bar", figsize=(11, 5), colormap="viridis")
        plt.title("TCP Flag Frequency by Attack Category (Proportion per Flow)", fontweight="bold")
        plt.xlabel("Attack Category")
        plt.ylabel("Mean Flag Activation per Flow")
        plt.xticks(rotation=25)
        plt.legend(title="TCP Flags")
        plt.tight_layout()
        fig4_path = FIGURES_DIR / "04_tcp_flags_by_attack.png"
        plt.savefig(fig4_path, dpi=300)
        plt.close()
        print(f"[+] Saved: {fig4_path}")

    # Figure 5: Feature Correlation Heatmap
    top_numeric = df_clean.select_dtypes(include=[np.number]).columns[:15]
    corr_matrix = df_clean[top_numeric].corr()
    plt.figure(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=0.5)
    plt.title("Correlation Matrix of Top Flow Telemetry Features", fontweight="bold")
    plt.tight_layout()
    fig5_path = FIGURES_DIR / "05_feature_correlation_heatmap.png"
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {fig5_path}")

    # 5. Export JSON Summary Report
    summary_data = {
        "dataset_name": dataset_name,
        "total_records": int(len(df)),
        "clean_records": int(len(df_clean)),
        "feature_count": int(df.shape[1]),
        "total_nulls": total_nulls,
        "total_infs": total_infs,
        "class_distribution": class_summary,
        "imbalance_ratios_to_majority": imbalance_ratios,
        "generated_figures": [
            str(fig1_path.name),
            str(fig2_path.name) if "Flow Duration" in df_clean.columns else None,
            str(fig3_path.name) if "Fwd Packet Length Mean" in df_clean.columns else None,
            str(fig4_path.name) if flag_cols else None,
            str(fig5_path.name)
        ]
    }
    
    summary_path = REPORTS_DIR / "eda_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=4)
    print(f"\n[+] Exported EDA Summary to: {summary_path}")
    print("=" * 70)
    return summary_data


if __name__ == "__main__":
    run_eda()
