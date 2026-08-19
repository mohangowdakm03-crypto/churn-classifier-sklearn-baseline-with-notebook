"""
Tests for the data loader module.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import validate


def make_sample_df(**overrides):
    base = {
        "customerID": ["C001"],
        "gender": ["Male"],
        "SeniorCitizen": [0],
        "Partner": ["Yes"],
        "Dependents": ["No"],
        "tenure": [12],
        "PhoneService": ["Yes"],
        "MultipleLines": ["No"],
        "InternetService": ["Fiber optic"],
        "OnlineSecurity": ["No"],
        "OnlineBackup": ["Yes"],
        "DeviceProtection": ["No"],
        "TechSupport": ["No"],
        "StreamingTV": ["No"],
        "StreamingMovies": ["No"],
        "Contract": ["Month-to-month"],
        "PaperlessBilling": ["Yes"],
        "PaymentMethod": ["Electronic check"],
        "MonthlyCharges": [65.0],
        "TotalCharges": ["780.0"],
        "Churn": ["No"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_validate_returns_dataframe():
    df = make_sample_df()
    result = validate(df)
    assert isinstance(result, pd.DataFrame)


def test_validate_converts_total_charges_to_numeric():
    df = make_sample_df()
    result = validate(df)
    assert pd.api.types.is_float_dtype(result["TotalCharges"])


def test_validate_drops_nan_total_charges():
    df = make_sample_df(TotalCharges=[" "])
    result = validate(df)
    assert len(result) == 0


def test_validate_raises_on_missing_columns():
    df = make_sample_df()
    df = df.drop(columns=["Churn"])
    with pytest.raises(ValueError, match="missing expected columns"):
        validate(df)


def test_validate_preserves_row_count_on_valid_data():
    rows = {k: [v[0]] * 10 for k, v in make_sample_df().to_dict(orient="list").items()}
    df = pd.DataFrame(rows)
    result = validate(df)
    assert len(result) == 10
