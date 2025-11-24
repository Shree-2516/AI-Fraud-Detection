# 💳 AI-Based Financial Fraud Detection System

An AI-powered system that detects fraudulent financial transactions using machine learning.  
Built with Python, Flask, XGBoost, SMOTE, and Streamlit.

## 🚀 Features
- Preprocessing and feature engineering
- Data balancing using SMOTE
- Model training (Random Forest, XGBoost)
- Real-time fraud prediction with Flask API
- Interactive Streamlit dashboard for users
- Explainability using SHAP 

## 🧠 Tech Stack
`Python`, `Flask`, `Streamlit`, `Scikit-learn`, `XGBoost`, `SHAP`, `SMOTE`

## 📂 Folder Structure

src/ – backend scripts
models/ – saved trained models
data/ – dataset (download from Kaggle)
notebooks/ – EDA and training notebooks


## ⚙️ Setup
```bash
git clone https://github.com/Shree-2516/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt
python src/api.py
streamlit run streamlit_app.py

🧩 Demo
Example input:

{"amount_log": 3.2, "hour": 15, "Amount": 120}

Output:
{"fraud_probability": 0.47, "fraud_flag": false}


📈 Results

Accuracy: ~99%

ROC-AUC: 0.99

Balanced data using SMOTE

Explainability via SHAP summary plots

🧾 Dataset

Credit Card Fraud Detection Dataset — From kaggle - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud


👨‍💻 Author

Shreeyash Paraj
