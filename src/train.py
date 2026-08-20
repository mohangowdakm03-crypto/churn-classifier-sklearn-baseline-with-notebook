"""Model training and cross-validated evaluation for the churn classifier.

Cross-validates LogisticRegression, RandomForestClassifier, and
GradientBoostingClassifier on ROC-AUC and F1, then retrains the best
model on the full training split and serialises it with joblib.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.preprocess import build_pipeline

MODELS: dict[str, Any] = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, random_state=42
    ),
}

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Cross-validate all candidate models and return a summary DataFrame.

    Parameters
    ----------
    X:
        Feature matrix (training split).
    y:
        Binary target (training split).

    Returns
    -------
    pd.DataFrame
        Rows indexed by model name; columns:
        ``roc_auc_mean``, ``roc_auc_std``, ``f1_mean``, ``f1_std``.
    """
    results = {}
    for name, clf in MODELS.items():
        pipe = build_pipeline(clf)
        scores = cross_validate(
            pipe,
            X,
            y,
            cv=CV,
            scoring=["roc_auc", "f1"],
            n_jobs=-1,
        )
        results[name] = {
            "roc_auc_mean": scores["test_roc_auc"].mean(),
            "roc_auc_std": scores["test_roc_auc"].std(),
            "f1_mean": scores["test_f1"].mean(),
            "f1_std": scores["test_f1"].std(),
        }
    return pd.DataFrame(results).T


def train_best_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
) -> Pipeline:
    """Train the chosen model on the full training split.

    Parameters
    ----------
    X:
        Feature matrix (full training split, not CV folds).
    y:
        Binary target.
    model_name:
        Key in ``MODELS`` — one of ``'LogisticRegression'``,
        ``'RandomForest'``, or ``'GradientBoosting'``.

    Returns
    -------
    Pipeline
        Fitted end-to-end pipeline ready for inference.
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(MODELS)}")
    clf = MODELS[model_name]
    pipe = build_pipeline(clf)
    pipe.fit(X, y)
    return pipe


def save_model(pipeline: Pipeline, path: str | Path) -> None:
    """Serialise a fitted pipeline to disk with joblib.

    Parameters
    ----------
    pipeline:
        Fitted sklearn Pipeline.
    path:
        Destination file path (e.g. ``models/churn_pipeline.joblib``).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
