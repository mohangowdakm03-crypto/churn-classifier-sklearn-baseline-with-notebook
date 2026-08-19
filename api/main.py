#!/usr/bin/env python3
"""
FastAPI application for serving churn predictions.

Endpoints:
  GET  /health        — liveness check
  GET  /metrics       — model metrics from last training run
  POST /predict       — predict churn probability for a single customer
  POST /predict/batch — predict churn for a list of customers
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
PIPELINE_PATH = MODELS_DIR / "preprocessor.pkl"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
THRESHOLD_PATH = MODELS_DIR / "best_threshold.pkl"
METRICS_PATH = MODELS_DIR / "reports" / "metrics.json"
RESULTS_PATH = MODELS_DIR / "results.json"
DB_PATH = BASE_DIR / "data" / "inference_log.db"

REPORTS_DIR = MODELS_DIR / "reports"

# ── SQLite Inference Log ───────────────────────────────────────────────────────

def _init_db():
    """Create the SQLite DB and inference_log table if they don't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inference_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                gender TEXT,
                senior_citizen INTEGER,
                partner TEXT,
                dependents TEXT,
                tenure REAL,
                internet_service TEXT,
                contract TEXT,
                payment_method TEXT,
                monthly_charges REAL,
                total_charges REAL,
                churn_probability REAL,
                will_churn INTEGER,
                confidence TEXT,
                model_version TEXT
            )
        """)
        conn.commit()
    logger.info("SQLite inference log initialised at %s", DB_PATH)


@contextmanager
def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _log_prediction(customer, response):
    """Persist one inference record to the SQLite database."""
    try:
        with _get_db() as conn:
            conn.execute("""
                INSERT INTO inference_log VALUES (
                    :id, :timestamp, :gender, :senior_citizen, :partner,
                    :dependents, :tenure, :internet_service, :contract,
                    :payment_method, :monthly_charges, :total_charges,
                    :churn_probability, :will_churn, :confidence, :model_version
                )
            """, {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "gender": customer.gender,
                "senior_citizen": customer.SeniorCitizen,
                "partner": customer.Partner,
                "dependents": customer.Dependents,
                "tenure": customer.tenure,
                "internet_service": customer.InternetService,
                "contract": customer.Contract,
                "payment_method": customer.PaymentMethod,
                "monthly_charges": customer.MonthlyCharges,
                "total_charges": customer.TotalCharges,
                "churn_probability": response.churn_probability,
                "will_churn": int(response.will_churn),
                "confidence": response.confidence,
                "model_version": response.model_version,
            })
    except Exception as exc:
        logger.warning("Failed to log prediction to DB: %s", exc)

# ── Load artefacts at startup ─────────────────────────────────────────────────
_pipeline = None
_model = None
_threshold = 0.5


def _load_artefacts():
    global _pipeline, _model, _threshold
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. "
            "Run `python scripts/train_pipeline.py` first."
        )
    _pipeline = joblib.load(PIPELINE_PATH)
    _model = joblib.load(MODEL_PATH)
    if THRESHOLD_PATH.exists():
        _threshold = float(joblib.load(THRESHOLD_PATH))
    logger.info("Loaded model from %s (threshold=%.3f)", MODEL_PATH, _threshold)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    model_config = {"json_schema_extra": {"example": {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 65.0, "TotalCharges": 780.0,
    }}}

    gender: str = Field(..., description="'Male' or 'Female'")
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(..., ge=0, le=72)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    churn_probability: float
    will_churn: bool
    confidence: str  # "high" | "medium" | "low"
    model_version: str


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int


# ── App ───────────────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app):
    _load_artefacts()
    _init_db()
    yield



app = FastAPI(
    title="Churn Classifier API",
    description=(
        "Production-ready REST API for predicting customer churn. "
        "Uses the best model selected from Logistic Regression, "
        "Random Forest, XGBoost, and LightGBM via Optuna HPO."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Serve the new premium frontend and static assets
frontend_path = BASE_DIR / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

if REPORTS_DIR.exists():
    app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")


@app.get("/", include_in_schema=False)
async def serve_frontend():
    index_file = frontend_path / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "API is running, but frontend not found."}





BINARY_MAP = {"Yes": 1, "No": 0, "No phone service": 0, "No internet service": 0}
BINARY_COLS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
ADDON_SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def _prepare_for_pipeline(customer: CustomerFeatures) -> pd.DataFrame:
    """
    Convert a CustomerFeatures object into a DataFrame that matches
    exactly the columns the fitted ColumnTransformer expects.
    """
    d = customer.model_dump()

    # Engineered features
    charges_ratio = d["MonthlyCharges"] / (d["TotalCharges"] + 1e-9)
    tenure = d["tenure"]
    if tenure <= 12:
        tenure_bucket = "0-1yr"
    elif tenure <= 24:
        tenure_bucket = "1-2yr"
    elif tenure <= 48:
        tenure_bucket = "2-4yr"
    else:
        tenure_bucket = "4+yr"
    service_count = sum(1 for k in ADDON_SERVICES if d.get(k) == "Yes")
    is_month_to_month = int(d["Contract"] == "Month-to-month")

    # Binary encode yes/no cols
    binary_encoded = {col: BINARY_MAP.get(d[col], 0) for col in BINARY_COLS}
    gender = int(d["gender"] == "Male")

    return pd.DataFrame([{
        # Numeric cols (StandardScaler)
        "SeniorCitizen": d["SeniorCitizen"],
        "tenure": d["tenure"],
        "MonthlyCharges": d["MonthlyCharges"],
        "TotalCharges": d["TotalCharges"],
        "charges_ratio": charges_ratio,
        # OHE cols
        "MultipleLines": d["MultipleLines"],
        "InternetService": d["InternetService"],
        "Contract": d["Contract"],
        "PaymentMethod": d["PaymentMethod"],
        "tenure_bucket": tenure_bucket,
        # Passthrough (binary)
        "gender": gender,
        **binary_encoded,
        "service_count": service_count,
        "is_month_to_month": is_month_to_month,
    }])


def _predict_single(customer: CustomerFeatures) -> PredictionResponse:
    row = _prepare_for_pipeline(customer)
    X = _pipeline.transform(row)
    prob = float(_model.predict_proba(X)[0, 1])
    will_churn = prob >= _threshold
    if prob >= 0.75 or prob <= 0.25:
        confidence = "high"
    elif prob >= 0.60 or prob <= 0.40:
        confidence = "medium"
    else:
        confidence = "low"
    return PredictionResponse(
        churn_probability=round(prob, 4),
        will_churn=will_churn,
        confidence=confidence,
        model_version=MODEL_PATH.stat().st_mtime.__str__()[:10],
    )


@app.get("/health", tags=["Health"])
def health():
    """Liveness check. Returns 200 if the model is loaded."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_loaded": True}


@app.get("/metrics", tags=["Model"])
def get_metrics():
    """Return the latest training metrics."""
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="No metrics found. Train the model first.")


@app.get("/results", tags=["Model"])
def get_results():
    """Return the full model comparison results."""
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="No results found. Train the model first.")


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerFeatures):
    """Predict churn probability for a single customer."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = _predict_single(customer)
        _log_prediction(customer, result)   # ← persist to SQLite
        return result
    except Exception as exc:
        logger.exception("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(request: BatchPredictionRequest):
    """Predict churn for a batch of up to 1000 customers."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(request.customers) > 1000:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 1000")
    try:
        predictions = [_predict_single(c) for c in request.customers]
        for customer, result in zip(request.customers, predictions):
            _log_prediction(customer, result)
        return BatchPredictionResponse(predictions=predictions, count=len(predictions))
    except Exception as exc:
        logger.exception("Batch prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/audit-log", tags=["Audit"])
def get_audit_log(
    limit: int = Query(50, ge=1, le=500, description="Max rows to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """
    Return paginated inference history from the SQLite audit log.
    Sorted newest-first.
    """
    if not DB_PATH.exists():
        return {"records": [], "total": 0}
    with _get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM inference_log").fetchone()[0]
        rows = conn.execute(
            """
            SELECT id, timestamp, gender, senior_citizen, partner, dependents,
                   tenure, internet_service, contract, payment_method,
                   monthly_charges, total_charges,
                   churn_probability, will_churn, confidence, model_version
            FROM inference_log
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
    return {
        "records": [dict(r) for r in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.delete("/audit-log", tags=["Audit"])
def clear_audit_log():
    """Clear all records from the inference audit log."""
    if not DB_PATH.exists():
        return {"deleted": 0}
    with _get_db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM inference_log").fetchone()[0]
        conn.execute("DELETE FROM inference_log")
    return {"deleted": count, "message": "Audit log cleared successfully."}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
