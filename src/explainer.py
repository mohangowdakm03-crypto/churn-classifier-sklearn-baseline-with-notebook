#!/usr/bin/env python3
"""
SHAP-based model explainability module.
Generates global and local SHAP explanations and saves artefacts.
Handles both 2D (linear/xgb) and 3D (RF TreeExplainer) shap_values shapes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)
REPORTS_DIR = Path(__file__).resolve().parent.parent / "models" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _to_2d(shap_values) -> np.ndarray:
    """
    Normalise SHAP values to 2D (samples × features).
    - Already 2D → return as-is.
    - 3D (samples × classes × features) → take class-1 slice.
    - list of arrays (old SHAP API) → take index-1 element.
    """
    if isinstance(shap_values, list):
        arr = np.array(shap_values[1])
    else:
        arr = np.array(shap_values)

    if arr.ndim == 3:
        # shape: (samples, features, classes) or (samples, classes, features)
        # Use the class-1 slice along whichever axis has size 2
        if arr.shape[1] == 2:
            arr = arr[:, 1, :]      # (samples, features)
        elif arr.shape[2] == 2:
            arr = arr[:, :, 1]      # (samples, features)
        else:
            arr = arr[:, :, 0]      # fallback: just take first
    return arr


def compute_shap_values(model: Any, X: np.ndarray, model_name: str = "model"):
    """Compute SHAP values using the most appropriate explainer."""
    import shap

    try:
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            raw = explainer.shap_values(X)
        else:
            explainer = shap.LinearExplainer(model, X)
            raw = explainer.shap_values(X)

        shap_values = _to_2d(raw)
        return shap_values, explainer
    except Exception as exc:
        logger.warning("Could not compute SHAP values: %s", exc)
        return None, None


def plot_shap_summary(shap_values, X, feature_names: List[str], save_path: Path = None):
    """Generate and save a SHAP beeswarm summary plot."""
    import shap

    if shap_values is None:
        logger.warning("No SHAP values available, skipping summary plot.")
        return

    path = save_path or REPORTS_DIR / "shap_summary.png"
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved SHAP summary plot to %s", path)


def plot_shap_bar(shap_values, feature_names: List[str], save_path: Path = None):
    """Generate and save a SHAP mean absolute bar chart."""
    if shap_values is None:
        return

    sv = _to_2d(shap_values)   # ensure 2D even if passed raw
    mean_abs = np.abs(sv).mean(axis=0)
    n_top = min(20, len(mean_abs))
    idx = np.argsort(mean_abs)[-n_top:]

    path = save_path or REPORTS_DIR / "shap_bar.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.arange(len(idx)), mean_abs[idx], align="center")
    ax.set_yticks(np.arange(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx], fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Impact (SHAP)")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved SHAP bar chart to %s", path)


def explain(
    model: Any,
    X_sample: np.ndarray,
    feature_names: List[str],
    model_name: str = "model",
):
    """Run full explainability pipeline."""
    shap_values, explainer = compute_shap_values(model, X_sample, model_name)
    plot_shap_summary(shap_values, X_sample, feature_names)
    plot_shap_bar(shap_values, feature_names)
    return shap_values, explainer
