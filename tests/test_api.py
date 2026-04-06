"""
tests/test_api.py
-----------------
API endpoint tests using FastAPI TestClient.
In production CI/CD: these run on every PR before deployment.

IB Engineering Standard:
- Unit tests for schema validation
- Integration tests for endpoint responses
- Mock the predictor for speed (don't call real model in CI)
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# ── Mock predictor before importing app ──────
MOCK_PREDICTION = {
    "signal"           : "LONG",
    "probability"      : 0.73,
    "confidence"       : "HIGH",
    "regime"           : "BULLISH",
    "real_yield_signal": "NEGATIVE (-0.45%) → Supportive for gold",
    "prediction_date"  : "2024-06-15",
    "model_version"    : "lgbm-v20240615",
    "features_used"    : 34,
    "threshold_used"   : 0.5,
    "disclaimer"       : "For research purposes only. Not financial advice.",
}


@pytest.fixture
def client():
    """Test client with mocked predictor."""
    with patch("api.predictor.predictor") as mock_pred:
        mock_pred.is_loaded     = True
        mock_pred.model_version = "lgbm-test-v1"
        mock_pred._last_pred_date = "2024-06-15"
        mock_pred.feature_cols  = [f"feat_{i}" for i in range(34)]
        mock_pred.metrics       = {"cv_auc_mean": 0.62, "cv_sharpe_mean": 1.4}
        mock_pred.predict.return_value = MOCK_PREDICTION
        mock_pred.get_feature_importance.return_value = [
            {"feature": "real_yield", "importance": 1500.0},
            {"feature": "dxy",        "importance": 1200.0},
        ]
        mock_pred.predict_batch.return_value = [
            {"date": "2024-06-10", "signal": "LONG",  "probability": 0.68,
             "actual": 1, "return_1d": 0.0123, "correct": True},
            {"date": "2024-06-11", "signal": "SHORT", "probability": 0.38,
             "actual": 0, "return_1d": -0.0045, "correct": True},
        ]

        from api.main import app
        with TestClient(app) as c:
            yield c


# ─────────────────────────────────────────────
# HEALTH TESTS
# ─────────────────────────────────────────────

def test_health_returns_200(client):
    r = client.get("/health")
    assert r.status_code == 200

def test_health_schema(client):
    r = client.get("/health")
    data = r.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "model_version" in data
    assert data["status"] == "healthy"

def test_root_endpoint(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "XAU/USD ML Signal API" in r.json()["service"]


# ─────────────────────────────────────────────
# MODEL INFO TESTS
# ─────────────────────────────────────────────

def test_model_info_returns_200(client):
    r = client.get("/model/info")
    assert r.status_code == 200

def test_model_info_schema(client):
    data = client.get("/model/info").json()
    assert "model_version" in data
    assert "n_features" in data
    assert "feature_names" in data
    assert "training_metrics" in data
    assert "top_features" in data
    assert data["n_features"] == 34


# ─────────────────────────────────────────────
# PREDICT TESTS
# ─────────────────────────────────────────────

def test_predict_returns_200(client):
    r = client.post("/predict", json={})
    assert r.status_code == 200

def test_predict_signal_valid(client):
    data = client.post("/predict", json={}).json()
    assert data["signal"] in ["LONG", "SHORT", "NEUTRAL"]

def test_predict_probability_range(client):
    data = client.post("/predict", json={}).json()
    assert 0.0 <= data["probability"] <= 1.0

def test_predict_confidence_valid(client):
    data = client.post("/predict", json={}).json()
    assert data["confidence"] in ["HIGH", "MEDIUM", "LOW"]

def test_predict_regime_valid(client):
    data = client.post("/predict", json={}).json()
    assert data["regime"] in ["BULLISH", "BEARISH", "NEUTRAL"]

def test_predict_with_date(client):
    r = client.post("/predict", json={"date": "2024-06-15"})
    assert r.status_code == 200

def test_predict_invalid_date(client):
    r = client.post("/predict", json={"date": "not-a-date"})
    assert r.status_code == 422  # Pydantic validation error

def test_predict_threshold_out_of_range(client):
    r = client.post("/predict", json={"threshold": 1.5})
    assert r.status_code == 422

def test_predict_has_disclaimer(client):
    data = client.post("/predict", json={}).json()
    assert "disclaimer" in data
    assert len(data["disclaimer"]) > 0


# ─────────────────────────────────────────────
# BATCH PREDICT TESTS
# ─────────────────────────────────────────────

def test_batch_predict_returns_200(client):
    r = client.post("/predict/batch", json={
        "start_date": "2024-06-10",
        "end_date"  : "2024-06-15",
    })
    assert r.status_code == 200

def test_batch_predict_schema(client):
    data = client.post("/predict/batch", json={
        "start_date": "2024-06-10",
        "end_date"  : "2024-06-15",
    }).json()
    assert "predictions" in data
    assert "n_predictions" in data
    assert "win_rate" in data
    assert data["n_predictions"] == 2

def test_batch_predict_win_rate_range(client):
    data = client.post("/predict/batch", json={
        "start_date": "2024-06-10",
        "end_date"  : "2024-06-15",
    }).json()
    if data["win_rate"] is not None:
        assert 0.0 <= data["win_rate"] <= 1.0


# ─────────────────────────────────────────────
# SCHEMA VALIDATION TESTS
# ─────────────────────────────────────────────

def test_predict_response_all_fields_present(client):
    data = client.post("/predict", json={}).json()
    required_fields = [
        "signal", "probability", "confidence", "regime",
        "prediction_date", "model_version", "features_used",
        "threshold_used", "disclaimer"
    ]
    for field in required_fields:
        assert field in data, f"Missing field: {field}"