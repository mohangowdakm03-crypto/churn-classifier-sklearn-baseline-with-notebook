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
    assert "threshold" in data


def test_predict_returns_valid_response(client):
    resp = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert resp.status_code == 200
    data = resp.json()
    assert "churn_probability" in data
    assert "will_churn" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert isinstance(data["will_churn"], bool)
    assert data["confidence"] in {"high", "medium", "low"}
    assert "model_version" in data


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
    assert resp.status_code == 422  # Pydantic max_length now catches this


def test_predict_batch_empty(client):
    """Empty batch should be rejected."""
    resp = client.post("/predict/batch", json={"customers": []})
    assert resp.status_code == 422


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    # Either 200 with metrics or 404 if not trained
    assert resp.status_code in {200, 404}


def test_invalid_input_negative_tenure(client):
    bad = {**SAMPLE_CUSTOMER, "tenure": -1}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422  # Pydantic validation error


def test_invalid_input_unknown_gender(client):
    """Unknown gender should be rejected with 422, not silently encoded."""
    bad = {**SAMPLE_CUSTOMER, "gender": "Unknown"}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_invalid_input_unknown_contract(client):
    """Unknown contract type should be rejected with 422."""
    bad = {**SAMPLE_CUSTOMER, "Contract": "Quarterly"}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_invalid_input_negative_charges(client):
    bad = {**SAMPLE_CUSTOMER, "MonthlyCharges": -10.0}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_audit_log_endpoint(client):
    """Audit log endpoint should return a valid paginated response."""
    resp = client.get("/audit-log")
    assert resp.status_code == 200
    data = resp.json()
    assert "records" in data
    assert "total" in data
    assert isinstance(data["records"], list)


def test_audit_log_pagination(client):
    """Pagination params should be respected."""
    resp = client.get("/audit-log?limit=5&offset=0")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["records"]) <= 5
    assert data["limit"] == 5


def test_audit_log_invalid_limit(client):
    """Limit > 500 should be rejected."""
    resp = client.get("/audit-log?limit=1000")
    assert resp.status_code == 422


def test_predict_then_audit_log_has_record(client):
    """After a prediction, the audit log should have at least 1 record."""
    client.post("/predict", json=SAMPLE_CUSTOMER)
    resp = client.get("/audit-log?limit=1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1
    assert len(data["records"]) == 1
    record = data["records"][0]
    assert "churn_probability" in record
    assert "will_churn" in record


def test_audit_log_clear(client):
    """DELETE /audit-log should clear all records and return count."""
    resp = client.delete("/audit-log")
    assert resp.status_code == 200
    data = resp.json()
    assert "deleted" in data
    # Verify it's actually empty now
    check = client.get("/audit-log")
    assert check.json()["total"] == 0
