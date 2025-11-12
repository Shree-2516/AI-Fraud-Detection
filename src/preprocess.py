# Handles dataset loading, scaling, and saving prepared data.
# src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

def preprocess_data(input_csv, output_npz):
    df = pd.read_csv(input_csv)
    df.rename(columns={'Class': 'is_fraud'}, inplace=True)
    df['amount_log'] = np.log1p(df['Amount'])
    df['hour'] = (df['Time'] // 3600) % 24
    features = ['amount_log', 'hour', 'Amount']
    target = 'is_fraud'
    X, y = df[features], df[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, '../models/scaler.joblib')
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_scaled, y)
    np.savez_compressed(output_npz, X_res=X_res, y_res=y_res)
    print("✅ Preprocessing complete.")

if __name__ == "__main__":
    preprocess_data('../data/creditcard.csv', '../data/processed_data.npz')
