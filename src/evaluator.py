#!/usr/bin/env python3
"""
Evaluator: generates a comprehensive suite of evaluation artefacts.
- Classification report
- Confusion matrix (png)
- ROC curve (png)
- Precision-Recall curve (png)
- Feature importance (png, if applicable)
- Threshold analysis (F1 vs decision threshold)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)
REPORTS_DIR = Path(__file__).resolve().parent.parent / "models" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(y_true, y_pred, save_path: Path = None) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    path = save_path or REPORTS_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", path)


def plot_roc_curve(y_true, y_prob, save_path: Path = None) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = save_path or REPORTS_DIR / "roc_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved ROC curve to %s", path)


def plot_pr_curve(y_true, y_prob, save_path: Path = None) -> None:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    path = save_path or REPORTS_DIR / "pr_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved PR curve to %s", path)


def plot_threshold_analysis(y_true, y_prob, save_path: Path = None) -> None:
    thresholds = np.linspace(0.1, 0.9, 80)
    f1_scores = [
        f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    best_t = thresholds[np.argmax(f1_scores)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, f1_scores, lw=2)
    ax.axvline(best_t, color="red", linestyle="--", label=f"Best threshold = {best_t:.2f}")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 vs Decision Threshold")
    ax.legend()
    plt.tight_layout()
    path = save_path or REPORTS_DIR / "threshold_analysis.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved threshold analysis to %s  (best threshold=%.2f)", path, best_t)
    return best_t


def plot_feature_importance(model, feature_names, top_n=20, save_path: Path = None) -> None:
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])

    if importances is None:
        logger.warning("Model does not expose feature importances, skipping plot.")
        return

    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(idx)), importances[idx], align="center")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    path = save_path or REPORTS_DIR / "feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved feature importance to %s", path)


def evaluate(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list | None = None,
) -> Dict:
    """Full evaluation suite. Returns a metrics dict."""
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred.astype(float)
    )

    report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])
    logger.info("\n%s", report)

    metrics = {
        "f1": float(f1_score(y_test, y_pred, average="weighted")),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "classification_report": report,
    }

    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)
    plot_pr_curve(y_test, y_prob)
    best_t = plot_threshold_analysis(y_test, y_prob)
    metrics["best_threshold"] = float(best_t)

    if feature_names:
        plot_feature_importance(model, feature_names)

    # Persist metrics
    metrics_path = REPORTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "classification_report"}, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    return metrics
