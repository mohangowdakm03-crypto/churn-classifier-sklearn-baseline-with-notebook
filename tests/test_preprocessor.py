"""
Tests for the preprocessor module.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocessor import preprocess, _engineer_features


def make_full_df(n=200):
    """Create a minimal valid DataFrame for testing."""
    rng = np.random.default_rng(42)
    contracts = ["Month-to-month", "One year", "Two year"]
    internet = ["Fiber optic", "DSL", "No"]
    payment = ["Electronic check", "Mailed check", "Bank transfer (automatic)",
               "Credit card (automatic)"]
    yes_no = ["Yes", "No"]

    return pd.DataFrame({
        "customerID": [f"C{i:04d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": rng.choice(yes_no, n),
        "Dependents": rng.choice(yes_no, n),
        "tenure": rng.integers(0, 73, n),
        "PhoneService": rng.choice(yes_no, n),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(internet, n),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
        "Contract": rng.choice(contracts, n),
        "PaperlessBilling": rng.choice(yes_no, n),
        "PaymentMethod": rng.choice(payment, n),
        "MonthlyCharges": rng.uniform(18, 120, n),
        "TotalCharges": rng.uniform(100, 8000, n),
        "Churn": rng.choice(yes_no, n, p=[0.73, 0.27]),
    })


def test_engineer_features_adds_new_columns():
    df = make_full_df()
    result = _engineer_features(df)
    for col in ["charges_ratio", "tenure_bucket", "service_count", "is_month_to_month"]:
        assert col in result.columns, f"Missing engineered column: {col}"


def test_preprocess_returns_correct_shapes():
    df = make_full_df(300)
    X_tr, X_te, y_tr, y_te, pipe = preprocess(df)
    assert abs(X_te.shape[0] - 60) <= 5  # ~20% of 300
    assert X_tr.shape[1] == X_te.shape[1]
    assert len(y_tr) == X_tr.shape[0]
    assert len(y_te) == X_te.shape[0]


def test_preprocess_pipeline_is_fitted():
    from sklearn.pipeline import Pipeline as SKPipeline
    df = make_full_df(200)
    _, _, _, _, pipe = preprocess(df)
    assert isinstance(pipe, SKPipeline)


def test_preprocess_no_nan_in_output():
    df = make_full_df(200)
    X_tr, X_te, _, _, _ = preprocess(df)
    assert not np.isnan(X_tr).any()
    assert not np.isnan(X_te).any()
