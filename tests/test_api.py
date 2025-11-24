import pytest
from unittest.mock import patch, MagicMock
import json
import numpy as np

def test_home(client):
    """Test the home endpoint."""
    response = client.get('/')
    assert response.status_code == 200
    assert response.json == {"message": "✅ Fraud Detection API is running"}

def test_favicon(client):
    """Test the favicon endpoint."""
    response = client.get('/favicon.ico')
    assert response.status_code == 204

@patch('src.api.scaler')
@patch('src.api.model')
def test_predict_legitimate(mock_model, mock_scaler, client):
    """Test prediction for legitimate transaction."""
    # Mock scaler
    mock_scaler.transform.return_value = [[0.5, 15, 120]]
    
    # Mock model
    mock_model.predict_proba.return_value = np.array([[0.9, 0.1]]) # 10% fraud probability
    
    data = {
        "amount_log": 3.2,
        "hour": 15,
        "Amount": 120
    }
    
    response = client.post('/predict', json=data)
    assert response.status_code == 200, f"Response: {response.json}"
    result = response.json
    assert result['fraud_probability'] == 0.1
    assert result['fraud_flag'] is False

@patch('src.api.scaler')
@patch('src.api.model')
def test_predict_fraud(mock_model, mock_scaler, client):
    """Test prediction for fraudulent transaction."""
    # Mock scaler
    mock_scaler.transform.return_value = [[0.5, 15, 120]]
    
    # Mock model
    mock_model.predict_proba.return_value = np.array([[0.1, 0.9]]) # 90% fraud probability
    
    data = {
        "amount_log": 3.2,
        "hour": 15,
        "Amount": 120
    }
    
    response = client.post('/predict', json=data)
    assert response.status_code == 200, f"Response: {response.json}"
    result = response.json
    assert result['fraud_probability'] == 0.9
    assert result['fraud_flag'] is True

def test_predict_missing_fields(client):
    """Test prediction with missing fields."""
    data = {
        "amount_log": 3.2
        # Missing hour and Amount
    }
    
    response = client.post('/predict', json=data)
    assert response.status_code == 400
    assert "Missing required fields" in response.json['error']

def test_predict_empty_body(client):
    """Test prediction with empty body."""
    response = client.post('/predict', json={})
    assert response.status_code == 400
    assert "Empty request body" in response.json['error']
