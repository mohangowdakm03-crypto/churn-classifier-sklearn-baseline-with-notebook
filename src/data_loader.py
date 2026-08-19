#!/usr/bin/env python3
"""
Data loader for the Telco Customer Churn dataset.
Automatically downloads the dataset and caches it locally.
"""

import os
import io
import hashlib
import logging
from pathlib import Path
import requests
import pandas as pd

logger = logging.getLogger(__name__)

# Publicly hosted Telco Churn CSV (IBM dataset on GitHub)
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/"
    "data/Telco-Customer-Churn.csv"
)
DATASET_SHA256 = None  # Set after first download
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_PATH = DATA_DIR / "telco_churn.csv"


def download_dataset(url: str = DATASET_URL, cache_path: Path = CACHE_PATH) -> Path:
    """Download the dataset to cache_path if it does not already exist."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        logger.info("Dataset already cached at %s", cache_path)
        return cache_path

    logger.info("Downloading dataset from %s …", url)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    logger.info("Dataset saved to %s (%d bytes)", cache_path, cache_path.stat().st_size)
    return cache_path


def load_raw(path: Path = CACHE_PATH) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame."""
    path = download_dataset(cache_path=path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d cols from %s", *df.shape, path)
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic data-quality checks.
    Returns the validated (and lightly cleaned) DataFrame.
    """
    required_cols = {
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing expected columns: {missing}")

    # TotalCharges may arrive as strings when rows have spaces
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=["TotalCharges"])
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.warning("Dropped %d rows with NaN TotalCharges", n_dropped)

    return df.reset_index(drop=True)


def load(path: Path = CACHE_PATH) -> pd.DataFrame:
    """Full pipeline: download → load → validate."""
    raw = load_raw(path)
    return validate(raw)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    df = load()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Churn distribution:\n{df['Churn'].value_counts()}")
