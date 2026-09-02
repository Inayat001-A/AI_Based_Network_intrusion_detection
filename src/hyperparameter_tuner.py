"""
Hyperparameter Optimization Module for AI-Based NIDS
Uses Optuna to optimize tree ensembles (XGBoost, LightGBM) for multi-class detection.
"""

import sys
import optuna
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Suppress verbose Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_xgboost(X, y, n_trials=15, cv=3):
    """
    Optimizes XGBoost hyper-parameters using Stratified CV and Macro F1 score.
    """
    print(f"[*] Running Optuna optimization for XGBoost ({n_trials} trials, {cv}-fold CV)...")
    
    # Subsample if dataset is large for rapid search
    if len(X) > 15000:
        sample_indices = np.random.choice(len(X), size=15000, replace=False)
        X_sub, y_sub = X.iloc[sample_indices], y.iloc[sample_indices]
    else:
        X_sub, y_sub = X, y

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=25),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "eval_metric": "mlogloss",
            "random_state": 42,
            "n_jobs": -1
        }
        
        clf = XGBClassifier(**params)
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_sub, y_sub, cv=skf, scoring="f1_macro", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    print(f"[+] Best XGBoost Macro F1: {study.best_value:.4f}")
    print(f"[+] Best Parameters: {study.best_params}")
    return study.best_params


def tune_lightgbm(X, y, n_trials=15, cv=3):
    """
    Optimizes LightGBM hyper-parameters.
    """
    print(f"[*] Running Optuna optimization for LightGBM ({n_trials} trials, {cv}-fold CV)...")
    
    if len(X) > 15000:
        sample_indices = np.random.choice(len(X), size=15000, replace=False)
        X_sub, y_sub = X.iloc[sample_indices], y.iloc[sample_indices]
    else:
        X_sub, y_sub = X, y

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=25),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1
        }
        
        clf = LGBMClassifier(**params)
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_sub, y_sub, cv=skf, scoring="f1_macro", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    print(f"[+] Best LightGBM Macro F1: {study.best_value:.4f}")
    print(f"[+] Best Parameters: {study.best_params}")
    return study.best_params
