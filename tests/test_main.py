# tests/test_main.py
# Tests for the API endpoints.

from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings  # import the actual loaded settings

client = TestClient(app)


def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_predict_without_api_key_fails():
    response = client.post("/api/v1/predict", json={})
    assert response.status_code in (401, 422)


def test_predict_with_valid_data():
    """A valid, correctly-authenticated request should return a real prediction."""
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 171.0,
    }
    response = client.post(
        "/api/v1/predict",
        json=sample_customer,
        headers={"x-api-key": settings.api_key},  # use whatever key is actually loaded
    )
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert isinstance(data["will_churn"], bool)
    assert data["risk_level"] in ("low", "medium", "high")