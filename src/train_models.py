# Handles model training via command line.
# src/train_models.py
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from pathlib import Path  # <-- ADD THIS IMPORT

def train_models():
    # Define the base directory (two levels up from src)
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # Construct paths using pathlib
    DATA_PATH = BASE_DIR / 'data' / 'processed_data.npz'  # <-- USE PATHLIB
    RF_MODEL_PATH = BASE_DIR / 'models' / 'rf_model.joblib'
    XGB_MODEL_PATH = BASE_DIR / 'models' / 'xgb_model.joblib'

    # Load processed data
    data = np.load(DATA_PATH)  # <-- USE CORRECTED PATH VARIABLE
    X_res, y_res = data['X_res'], data['y_res']
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=42)

    # ... (Rest of the code remains similar, but update joblib paths)

    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:,1]
    print("RF AUC:", roc_auc_score(y_test, rf_probs))
    joblib.dump(rf, RF_MODEL_PATH)  # <-- USE PATH VARIABLE

    # ... (XGBoost training code)

    xgb_model = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    xgb_probs = xgb_model.predict_proba(X_test)[:,1]
    print("XGB AUC:", roc_auc_score(y_test, xgb_probs))
    joblib.dump(xgb_model, XGB_MODEL_PATH)  # <-- USE PATH VARIABLE

if __name__ == "__main__":
    train_models()