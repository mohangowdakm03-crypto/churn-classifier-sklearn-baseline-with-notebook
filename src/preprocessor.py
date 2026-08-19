#!/usr/bin/env python3
"""
Feature engineering and preprocessing pipeline for the Telco Churn dataset.
Uses sklearn Pipelines + ColumnTransformer for clean, production-safe design.

Note on class imbalance: We do NOT use SMOTE here. Class imbalance is handled
by class_weight='balanced' inside each model. SMOTE was causing val-set
threshold calibration to fail (perfectly balanced val → near-zero threshold
→ bad precision on real imbalanced test set).
"""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

DROP_COLS = ["customerID"]
TARGET_COL = "Churn"
BINARY_YES_NO = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
BINARY_MAP = {"Yes": 1, "No": 0, "No phone service": 0, "No internet service": 0}


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that boost model signal."""
    df = df.copy()
    df["charges_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1e-9)
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4+yr"],
    ).astype(str)
    addon_services = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["service_count"] = df[addon_services].apply(
        lambda row: sum(1 for v in row if v == "Yes"), axis=1
    )
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    return df


def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Pipeline]:
    """
    Full preprocessing: engineer → encode → split → scale/OHE.
    Returns (X_train, X_test, y_train, y_test, fitted_pipeline).
    Class imbalance is handled by class_weight inside each model.
    """
    df = _engineer_features(df)
    y = (df[TARGET_COL] == "Yes").astype(int).values
    df = df.drop(columns=DROP_COLS + [TARGET_COL])

    for col in BINARY_YES_NO:
        if col in df.columns:
            df[col] = df[col].map(BINARY_MAP).fillna(0).astype(int)
    df["gender"] = (df["gender"] == "Male").astype(int)

    passthrough_cols = [
        c for c in df.columns
        if c in BINARY_YES_NO + ["gender", "is_month_to_month", "service_count"]
    ]
    ohe_cols = [
        c for c in df.columns
        if df[c].dtype == object or c == "tenure_bucket"
    ]
    numeric_cols = [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in passthrough_cols
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ohe_cols),
            ("pass", "passthrough", passthrough_cols),
        ],
        remainder="drop",
    )
    pipeline = Pipeline([("preprocessor", preprocessor)])

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        df, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    X_train = pipeline.fit_transform(X_train_df)
    X_test = pipeline.transform(X_test_df)

    logger.info(
        "Preprocessed: train=%d (churn=%.1f%%), test=%d (churn=%.1f%%), features=%d",
        len(y_train), 100 * y_train.mean(),
        len(y_test), 100 * y_test.mean(),
        X_train.shape[1],
    )
    return X_train, X_test, y_train, y_test, pipeline


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    from src.data_loader import load
    df = load()
    X_tr, X_te, y_tr, y_te, pipe = preprocess(df)
    print(f"X_train: {X_tr.shape}, X_test: {X_te.shape}")
    print(f"Train churn rate: {y_tr.mean():.2%}, Test churn rate: {y_te.mean():.2%}")
