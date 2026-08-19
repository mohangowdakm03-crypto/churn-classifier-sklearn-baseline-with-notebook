#!/usr/bin/env python3
"""
Multi-model trainer with Optuna hyperparameter optimisation.
Trains Logistic Regression, Random Forest, XGBoost, and LightGBM.

Design choices:
- class_weight='balanced' handles class imbalance instead of SMOTE (avoids
  threshold calibration failure on resampled val sets).
- Threshold is tuned on a stratified val set to maximise F1 on the real
  imbalanced distribution.
- Optuna tunes on F1@best_threshold so it directly optimises what we care about.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find the probability threshold that maximises F1 on the given split."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precision * recall / (precision + recall + 1e-9)
    # thresholds is 1 shorter than precision/recall
    best_idx = int(np.argmax(f1s[:-1]))
    return float(thresholds[best_idx])


# ── Builder functions (all use class_weight='balanced') ───────────────────────

def _build_lr(params: Dict) -> LogisticRegression:
    return LogisticRegression(
        **params, class_weight="balanced", max_iter=1000, random_state=42
    )


def _build_rf(params: Dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        **params, class_weight="balanced", random_state=42, n_jobs=-1
    )


def _build_xgb(params: Dict, scale_pos_weight: float = 1.0) -> XGBClassifier:
    return XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )


def _build_lgbm(params: Dict) -> LGBMClassifier:
    return LGBMClassifier(
        **params, class_weight="balanced", random_state=42, verbose=-1
    )


# ── Optuna objective factories ────────────────────────────────────────────────

def _make_objective(model_name: str, X_tr, y_tr, X_val, y_val):
    neg, pos = np.bincount(y_tr)
    spw = float(neg) / float(pos)

    def objective(trial):
        if model_name == "logistic_regression":
            params = dict(
                C=trial.suggest_float("C", 1e-3, 10.0, log=True),
                solver=trial.suggest_categorical("solver", ["liblinear", "saga"]),
            )
            clf = _build_lr(params)
        elif model_name == "random_forest":
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
                max_depth=trial.suggest_int("max_depth", 4, 20),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
                max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            )
            clf = _build_rf(params)
        elif model_name == "xgboost":
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
                max_depth=trial.suggest_int("max_depth", 3, 8),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            )
            clf = _build_xgb(params, scale_pos_weight=spw)
        elif model_name == "lightgbm":
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
                max_depth=trial.suggest_int("max_depth", 3, 12),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                num_leaves=trial.suggest_int("num_leaves", 20, 80),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
            )
            clf = _build_lgbm(params)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        clf.fit(X_tr, y_tr)
        y_prob = clf.predict_proba(X_val)[:, 1]
        t = _best_threshold(y_val, y_prob)
        return f1_score(y_val, (y_prob >= t).astype(int))

    return objective


def optimise(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str,
    n_trials: int = 30,
) -> Tuple[Any, Dict, float]:
    """
    Run Optuna HPO on the (X_train, y_train) → evaluate on (X_val, y_val).
    Returns (model_retrained_on_all_data, best_params, optimal_threshold).
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(
        _make_objective(model_name, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_params = study.best_params
    logger.info("[%s] Best val-F1=%.4f  params=%s", model_name, study.best_value, best_params)

    # Re-train on full data (train + val) with best params
    X_all = np.vstack([X_train, X_val])
    y_all = np.hstack([y_train, y_val])
    neg, pos = np.bincount(y_all)
    spw = float(neg) / float(pos)

    if model_name == "logistic_regression":
        final = _build_lr(best_params)
    elif model_name == "random_forest":
        final = _build_rf(best_params)
    elif model_name == "xgboost":
        final = _build_xgb(best_params, scale_pos_weight=spw)
    elif model_name == "lightgbm":
        final = _build_lgbm(best_params)
    else:
        raise ValueError

    final.fit(X_all, y_all)

    # Calibrate threshold on val set (real imbalanced distribution)
    y_prob_val = final.predict_proba(X_val)[:, 1]
    threshold = _best_threshold(y_val, y_prob_val)

    return final, best_params, threshold


def train_all(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_trials: int = 20,
    val_fraction: float = 0.20,
) -> Tuple[List[Dict], str, Any, float]:
    """
    Train all 4 models with HPO.
    Uses a stratified val split so class ratios are preserved.
    Returns (results, best_model_name, best_model, best_threshold).
    """
    # Stratified split to preserve ~26% churn rate in val set
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=42)
    train_idx, val_idx = next(sss.split(X_train, y_train))
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    logger.info(
        "Val split: %d rows (churn=%.1f%%)", len(y_val), 100 * y_val.mean()
    )

    results = []
    model_names = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]

    for model_name in model_names:
        logger.info("── Optimising %s (%d trials) ──", model_name, n_trials)
        t0 = time.time()
        model, params, threshold = optimise(X_tr, y_tr, X_val, y_val, model_name, n_trials)
        elapsed = time.time() - t0

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        record = {
            "model": model_name,
            "f1": float(f1_score(y_test, y_pred, average="weighted")),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "threshold": round(threshold, 4),
            "best_params": params,
            "train_time_sec": round(elapsed, 2),
        }
        results.append(record)
        logger.info(
            "[%s] F1=%.4f  ROC-AUC=%.4f  threshold=%.3f  (%.1fs)",
            model_name, record["f1"], record["roc_auc"], threshold, elapsed,
        )

        joblib.dump(model, MODELS_DIR / f"{model_name}.pkl")
        joblib.dump(threshold, MODELS_DIR / f"{model_name}_threshold.pkl")

    best = max(results, key=lambda r: r["f1"])
    best_name = best["model"]
    best_threshold = best["threshold"]
    best_model = joblib.load(MODELS_DIR / f"{best_name}.pkl")

    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    joblib.dump(best_threshold, MODELS_DIR / "best_threshold.pkl")

    with open(MODELS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Best model: %s (F1=%.4f)", best_name, best["f1"])
    return results, best_name, best_model, best_threshold


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    from src.data_loader import load
    from src.preprocessor import preprocess

    df = load()
    X_tr, X_te, y_tr, y_te, pipe = preprocess(df)
    joblib.dump(pipe, MODELS_DIR / "preprocessor.pkl")

    results, best_name, best_model, best_t = train_all(X_tr, y_tr, X_te, y_te)
    print("\nModel Comparison:")
    for r in sorted(results, key=lambda x: x["f1"], reverse=True):
        print(f"  {r['model']:25s}  F1={r['f1']:.4f}  ROC-AUC={r['roc_auc']:.4f}  t={r['threshold']:.3f}")
    print(f"\nBest: {best_name}")
