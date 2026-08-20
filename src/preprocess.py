"""Preprocessing pipeline for the Telco churn dataset.

Builds a ColumnTransformer that handles numeric imputation + scaling
and categorical imputation + one-hot encoding. Wraps it in a full
sklearn Pipeline ready for a classifier head.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Column definitions for the Telco Customer Churn dataset
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

TARGET_COLUMN = "Churn"


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load and minimally clean the raw Telco churn CSV.

    Reads the CSV at *path*, coerces 'TotalCharges' to float (the raw
    file stores empty strings for customers with zero tenure), drops
    the non-informative 'customerID' column, and splits features from
    the binary target.

    Parameters
    ----------
    path:
        Filesystem path to the raw CSV (e.g. ``data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv``).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with shape (n_samples, n_features).
    y : pd.Series
        Binary target — 1 for churned, 0 for retained.
    """
    df = pd.read_csv(path)

    # customerID carries no signal
    df = df.drop(columns=["customerID"], errors="ignore")

    # TotalCharges is read as object when the field is blank
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Encode target
    df[TARGET_COLUMN] = (df[TARGET_COLUMN].str.strip().str.lower() == "yes").astype(int)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Build the feature preprocessing ColumnTransformer.

    Numeric columns: median imputation → standard scaling.
    Categorical columns: most-frequent imputation → one-hot encoding
    (unknown categories at inference time are ignored, not an error).

    Returns
    -------
    ColumnTransformer
        Unfitted transformer; call ``.fit_transform(X_train)`` or let
        a wrapping Pipeline handle fitting.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def build_pipeline(classifier) -> Pipeline:
    """Wrap the preprocessor and a classifier into a single sklearn Pipeline.

    Parameters
    ----------
    classifier:
        Any sklearn-compatible estimator (LogisticRegression,
        RandomForestClassifier, GradientBoostingClassifier, …).

    Returns
    -------
    Pipeline
        Unfitted end-to-end pipeline: preprocessor → classifier.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )
