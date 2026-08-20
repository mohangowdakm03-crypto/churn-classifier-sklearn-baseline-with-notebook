"""Inference helpers for the serialised churn pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def load_model(path: str | Path) -> Pipeline:
    """Load a joblib-serialised pipeline from disk.

    Parameters
    ----------
    path:
        Path to a ``.joblib`` file produced by ``train.save_model``.

    Returns
    -------
    Pipeline
        Fitted sklearn Pipeline.
    """
    return joblib.load(path)


def predict_proba(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return churn probability for each row in *X*.

    Parameters
    ----------
    pipeline:
        Fitted pipeline from ``load_model``.
    X:
        Feature DataFrame; must contain the columns the pipeline was
        trained on. Extra columns are ignored by the preprocessor.

    Returns
    -------
    np.ndarray
        1-D array of shape (n_samples,) with churn probabilities in [0, 1].
    """
    proba = pipeline.predict_proba(X)
    return proba[:, 1]


def predict_label(
    pipeline: Pipeline,
    X: pd.DataFrame,
    threshold: float = 0.5,
) -> np.ndarray:
    """Return binary churn label using a configurable probability threshold.

    Parameters
    ----------
    pipeline:
        Fitted pipeline.
    X:
        Feature DataFrame.
    threshold:
        Decision boundary. Rows with churn probability >= threshold are
        labelled 1 (churned). Default is 0.5.

    Returns
    -------
    np.ndarray
        1-D integer array of shape (n_samples,) with values 0 or 1.
    """
    return (predict_proba(pipeline, X) >= threshold).astype(int)
