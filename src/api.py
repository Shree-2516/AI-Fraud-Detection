# src/api.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SCALER_PATH = BASE_DIR / 'models' / 'scaler.joblib'
MODEL_PATH = BASE_DIR / 'models' / 'xgb_model.joblib'

# Features used during training
FEATURES = ['amount_log', 'hour', 'Amount']

# -----------------------------------------------------------
# Initialize Flask App
# -----------------------------------------------------------
app = Flask(__name__)

# Load scaler and model safely
try:
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)
    print("✅ Models loaded successfully")
except FileNotFoundError as e:
    print("❌ Model files not found:", e)
    scaler = None
    model = None

# -----------------------------------------------------------
# Routes
# -----------------------------------------------------------
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "✅ Fraud Detection API is running"})


@app.route('/predict', methods=['POST'])
def predict():
    # Check if model is loaded
    if scaler is None or model is None:
        return jsonify({"error": "Model files not found. Please run preprocessing and training first."}), 500

    data = request.get_json()

    # Validate input JSON
    if not data:
        return jsonify({"error": "Empty request body"}), 400

    # Check if all required features exist
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    # Prepare input
    try:
        X = pd.DataFrame([data])[FEATURES]
        X_scaled = scaler.transform(X)
        proba = model.predict_proba(X_scaled)[0, 1]
        result = {
            "fraud_probability": float(proba),
            "fraud_flag": bool(proba > 0.5)
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/favicon.ico')
def favicon():
    return '', 204


# -----------------------------------------------------------
# Run Server
# -----------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, port=5000)
