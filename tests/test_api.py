from fastapi.testclient import TestClient
from app import app
import pytest

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    # Match the message we just added to app.py
    assert response.json()["message"] == "Fraud Detection API is live!"

def test_prediction_legit():
    """Test a valid prediction request"""
    # Using 29 dummy features
    payload = {"features": [0.0] * 29}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    assert "is_fraud" in response.json()
    assert "explanation" in response.json()

def test_invalid_input():
    """Test if the API correctly rejects wrong number of features"""
    payload = {"features": [0.0, 1.1]} # Only 2 features instead of 29
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 400
    assert response.json()["detail"] == "Expected 29 features"