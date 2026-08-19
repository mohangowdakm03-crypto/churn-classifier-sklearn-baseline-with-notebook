#!/usr/bin/env python3
"""
train_pipeline.py — One-shot script to run the full training pipeline.

Usage:
    python scripts/train_pipeline.py [--n-trials N]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load
from src.preprocessor import preprocess
from src.trainer import train_all, MODELS_DIR
from src.evaluator import evaluate
from src.explainer import explain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(n_trials: int = 20):
    logger.info("════════════════════════════════════════")
    logger.info("  Churn Classifier — Training Pipeline  ")
    logger.info("════════════════════════════════════════")

    # 1. Load & preprocess
    logger.info("[1/5] Loading dataset …")
    df = load()
    logger.info("[2/5] Preprocessing …")
    X_train, X_test, y_train, y_test, pipeline = preprocess(df)
    joblib.dump(pipeline, MODELS_DIR / "preprocessor.pkl")
    logger.info("Preprocessor saved.")

    # Get feature names from the pipeline
    try:
        ohe_names = list(
            pipeline.named_steps["preprocessor"]
            .named_transformers_["ohe"]
            .get_feature_names_out()
        )
        num_names = list(
            pipeline.named_steps["preprocessor"]
            .transformers_[0][2]  # numeric cols
        )
        pass_names = list(
            pipeline.named_steps["preprocessor"]
            .transformers_[2][2]  # passthrough cols
        )
        feature_names = num_names + ohe_names + pass_names
    except Exception:
        feature_names = [f"f{i}" for i in range(X_test.shape[1])]

    # 2. Train all models with HPO
    logger.info("[3/5] Training all models (n_trials=%d per model) …", n_trials)
    results, best_name, best_model, best_threshold = train_all(
        X_train, y_train, X_test, y_test, n_trials=n_trials
    )
    joblib.dump(best_threshold, MODELS_DIR / "best_threshold.pkl")

    # 3. Evaluate best model
    logger.info("[4/5] Evaluating best model (%s) …", best_name)
    metrics = evaluate(best_model, X_test, y_test, feature_names=feature_names)

    # 4. SHAP explanation (on a sample of 200 rows)
    logger.info("[5/5] Generating SHAP explanations …")
    sample_size = min(200, len(X_test))
    explain(
        best_model,
        X_test[:sample_size],
        feature_names=feature_names,
        model_name=best_name,
    )

    # 5. Print summary
    print("\n" + "═" * 50)
    print("  MODEL COMPARISON")
    print("═" * 50)
    print(f"{'Model':25s}  {'F1':>6}  {'ROC-AUC':>8}")
    print("─" * 50)
    for r in sorted(results, key=lambda x: x["f1"], reverse=True):
        marker = "★" if r["model"] == best_name else " "
        print(f"{marker} {r['model']:23s}  {r['f1']:.4f}  {r['roc_auc']:.4f}")
    print("═" * 50)
    print(f"\nBest model : {best_name}")
    print(f"F1 Score   : {metrics['f1']:.4f}")
    print(f"ROC-AUC    : {metrics['roc_auc']:.4f}")
    print(f"\nArtifacts saved to: {MODELS_DIR}")

    target_f1 = 0.75
    if metrics["f1"] < target_f1:
        logger.warning(
            "⚠  F1=%.4f is below target of %.2f. Consider more trials or data.",
            metrics["f1"], target_f1,
        )
        return 1
    else:
        logger.info("✓ F1=%.4f exceeds target of %.2f.", metrics["f1"], target_f1)
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=20)
    args = parser.parse_args()
    sys.exit(main(n_trials=args.n_trials))
