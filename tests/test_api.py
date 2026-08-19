"""
Integration tests for the FastAPI application.
Requires the model to be trained first.
"""

import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

MODELS_DIR = BASE_DIR / "models"
MODEL_EXISTS = (MODELS_DIR / "best_model.pkl").exists()

SAMPLE_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 65.0,
    "TotalCharges": 780.0,
}


@pytest.fixture(scope="module")
def client():
    if not MODEL_EXISTS:
        pytest.skip("Model not trained yet — run scripts/train_pipeline.py first")
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_returns_valid_response(client):
    resp = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert resp.status_code == 200
    data = resp.json()
    assert "churn_probability" in data
    assert "will_churn" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert isinstance(data["will_churn"], bool)
    assert data["confidence"] in {"high", "medium", "low"}


def test_predict_batch(client):
    resp = client.post("/predict/batch", json={"customers": [SAMPLE_CUSTOMER] * 3})
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 3
    assert len(data["predictions"]) == 3


def test_predict_batch_limit(client):
    resp = client.post(
        "/predict/batch",
        json={"customers": [SAMPLE_CUSTOMER] * 1001},
    )
    assert resp.status_code == 400


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    # Either 200 with metrics or 404 if not trained
    assert resp.status_code in {200, 404}


def test_invalid_input(client):
    bad = {**SAMPLE_CUSTOMER, "tenure": -1}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422  # Pydantic validation error
