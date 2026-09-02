"""
Data Preprocessing and Feature Engineering Pipeline for AI-Based NIDS
Includes robust scaling, label encoding, feature selection, and artifact serialization.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


class NIDSPreprocessor:
    def __init__(self, top_k_features=30, scaler_type="robust"):
        self.top_k_features = top_k_features
        self.scaler_type = scaler_type
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        self.selected_features = []
        self.feature_importance_scores = {}
        self.fitted = False

    def clean_data(self, df):
        """
        Cleans infinite values, division-by-zero, and handles NaNs.
        """
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        # Replace positive and negative infinity with NaN, then forward fill / median impute
        for col in numeric_cols:
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val if not np.isnan(median_val) else 0)
                
        return df_clean

    def fit(self, df, label_col="Label"):
        """
        Fits the label encoder, selects top-k features, and fits the scaler on training data.
        """
        print(f"[*] Fitting NIDS Preprocessor on {len(df):,} records...")
        df_clean = self.clean_data(df)
        
        # 1. Fit Label Encoder
        if label_col not in df_clean.columns:
            raise ValueError(f"Target column '{label_col}' not found in dataframe.")
        y_encoded = self.label_encoder.fit_transform(df_clean[label_col])
        y = pd.Series(y_encoded, index=df_clean.index)
        
        # 2. Extract numeric feature candidates
        X = df_clean.drop(columns=[label_col]).select_dtypes(include=[np.number])
        
        # 3. Feature Selection using ExtraTrees Importance
        sample_size = min(15000, len(X))
        print(f"[*] Selecting top {self.top_k_features} flow features on {sample_size:,} samples...")
        X_sample = X.sample(n=sample_size, random_state=42)
        y_sample = y.loc[X_sample.index]
        
        selector = ExtraTreesClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
        selector.fit(X_sample, y_sample)
        
        importances = selector.feature_importances_
        feature_ranking = pd.Series(importances, index=X.columns).sort_values(ascending=False)
        
        k = min(self.top_k_features, len(feature_ranking))
        self.selected_features = list(feature_ranking.head(k).index)
        self.feature_importance_scores = {feat: float(score) for feat, score in feature_ranking.head(k).items()}
        
        # 4. Fit Scaler on selected features
        X_selected = X[self.selected_features]
        self.scaler.fit(X_selected)
        self.fitted = True
        
        print(f"[+] Selected top {len(self.selected_features)} features.")
        return self

    def transform(self, df, label_col="Label"):
        """
        Transforms input data using the fitted feature selector and scaler.
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")
            
        df_clean = self.clean_data(df)
        
        # Verify required features exist
        missing = [f for f in self.selected_features if f not in df_clean.columns]
        if missing:
            raise ValueError(f"Missing required features in transform data: {missing}")
            
        X_selected = df_clean[self.selected_features]
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_selected),
            columns=self.selected_features,
            index=df_clean.index
        )
        
        if label_col in df_clean.columns:
            y = self.label_encoder.transform(df_clean[label_col])
            X_scaled["Target"] = y
            
        return X_scaled

    def fit_transform(self, df, label_col="Label"):
        return self.fit(df, label_col=label_col).transform(df, label_col=label_col)

    def inverse_transform_labels(self, y_numeric):
        """Converts numeric predictions back to original string labels."""
        return self.label_encoder.inverse_transform(y_numeric)

    def save(self, models_dir=MODELS_DIR):
        """Serializes the fitted preprocessor and feature list to disk."""
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        
        artifact_path = models_dir / "preprocessor.joblib"
        features_path = models_dir / "selected_features.json"
        
        joblib.dump(self, artifact_path)
        with open(features_path, "w") as f:
            json.dump({
                "top_k": self.top_k_features,
                "selected_features": self.selected_features,
                "classes": list(self.label_encoder.classes_),
                "feature_importance_scores": self.feature_importance_scores
            }, f, indent=4)
            
        print(f"[+] Serialized preprocessor artifact to: {artifact_path}")
        print(f"[+] Saved selected features list to: {features_path}")

    @classmethod
    def load(cls, models_dir=MODELS_DIR):
        """Loads a pre-fitted NIDSPreprocessor from disk."""
        artifact_path = Path(models_dir) / "preprocessor.joblib"
        if not artifact_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {artifact_path}")
        return joblib.load(artifact_path)


def process_and_partition_dataset(input_csv, output_dir=PROCESSED_DIR, models_dir=MODELS_DIR, test_size=0.20, top_k=30):
    output_dir = Path(output_dir)
    models_dir = Path(models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("       NIDS PHASE 2: DATA PREPROCESSING & PARTITIONING")
    print("=" * 70)
    
    # 1. Load Raw CSV
    print(f"[*] Ingesting raw dataset from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"[+] Loaded {len(df):,} raw records with {df.shape[1]} columns.")
    
    label_col = "Label" if "Label" in df.columns else "label"
    
    # 2. Stratified Train / Test Split before fitting preprocessor (preventing data leakage)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df[label_col]
    )
    print(f"[*] Partitioned into {len(train_df):,} Train rows and {len(test_df):,} Test rows (Stratified).")
    
    # 3. Fit on Train only
    preprocessor = NIDSPreprocessor(top_k_features=top_k)
    train_processed = preprocessor.fit_transform(train_df, label_col=label_col)
    test_processed = preprocessor.transform(test_df, label_col=label_col)
    
    # 4. Save Processed Datasets
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    train_processed.to_csv(train_path, index=False)
    test_processed.to_csv(test_path, index=False)
    print(f"[+] Saved processed Train set: {train_path} (Shape: {train_processed.shape})")
    print(f"[+] Saved processed Test set:  {test_path} (Shape: {test_processed.shape})")
    
    # 5. Serialize Preprocessor Artifact
    preprocessor.save(models_dir=models_dir)
    print("=" * 70)
    return preprocessor, train_processed, test_processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NIDS Preprocessing Pipeline")
    parser.add_argument("--input", type=str, default=str(Path(__file__).resolve().parent.parent / "data" / "raw" / "cic_ids2017_benchmark.csv"))
    parser.add_argument("--output", type=str, default=str(PROCESSED_DIR))
    parser.add_argument("--models", type=str, default=str(MODELS_DIR))
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()
    
    process_and_partition_dataset(
        input_csv=args.input,
        output_dir=args.output,
        models_dir=args.models,
        top_k=args.top_k
    )
