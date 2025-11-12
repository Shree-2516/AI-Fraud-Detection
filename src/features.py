# You can define additional feature engineering functions here.
# src/features.py
import pandas as pd
import numpy as np

def add_transaction_features(df):
    df['amount_log'] = np.log1p(df['Amount'])
    df['hour'] = (df['Time'] // 3600) % 24
    return df
