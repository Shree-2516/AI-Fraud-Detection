# streamlit_app.py
import streamlit as st
import requests
import math
import os

st.set_page_config(page_title="AI Fraud Detection Dashboard", layout="centered")

st.title("💳 AI-Based Financial Fraud Detection System")
st.markdown("This app sends data to your Flask API running on **http://127.0.0.1:5000** and shows prediction results.")

# Input fields
st.subheader("Enter Transaction Details")

amount = st.number_input("Transaction Amount (₹)", min_value=0.0, step=0.1)
hour = st.number_input("Transaction Hour (0-23)", min_value=0, max_value=23, step=1)

# Calculate amount_log automatically
amount_log = math.log1p(amount) if amount > 0 else 0.0

st.write(f"💡 Log Amount (auto): {round(amount_log, 4)}")

# Prepare input JSON for API
data = {
    "amount_log": amount_log,
    "hour": hour,
    "Amount": amount
}

st.write("### Input JSON:", data)

# Predict button
if st.button("🔍 Predict Fraud"):
    try:
        api_url = os.getenv("API_URL", "http://127.0.0.1:5000")
        resp = requests.post(f"{api_url}/predict", json=data)
        if resp.status_code == 200:
            result = resp.json()
            prob = result["fraud_probability"]
            flag = result["fraud_flag"]

            st.success(f"Fraud Probability: **{prob*100:.2f}%**")
            if flag:
                st.error("🚨 Transaction is **Fraudulent!**")
            else:
                st.info("✅ Transaction appears **Legitimate**.")
        else:
            st.error(f"API Error {resp.status_code}: {resp.text}")
    except Exception as e:
        st.error(f"⚠️ Unable to connect to API: {e}")

st.markdown("---")
st.caption("© 2025 AI Fraud Detection System | Built with Streamlit + Flask + XGBoost")
