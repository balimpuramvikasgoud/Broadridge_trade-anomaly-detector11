import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Streamlit page config
st.set_page_config(page_title="Trade Anomaly Detector", layout="wide")
st.title("📈 Trade Anomaly Detection & Explanation")

# Paths to model and scaler
MODEL_PATH = "C:/Users/Lenovo/OneDrive/Desktop/4-1 P PROJECT/models/isolation_forest_model.pkl"
SCALER_PATH = "C:/Users/Lenovo/OneDrive/Desktop/4-1 P PROJECT/models/scaler.pkl"

# Load model & scaler
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("❌ Model or scaler file not found.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# File uploader
uploaded_file = st.file_uploader("📤 Upload trade CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # ✅ Feature Engineering (must match training pipeline)
        df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='min')
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['trade_type'] = np.random.choice(['BUY', 'SELL'], size=len(df))
        df['rolling_avg_volume'] = df['volume_vs_avg'].rolling(window=10, min_periods=1).mean()
        df['trader_id'] = np.random.randint(1000, 1010, size=len(df))
        trader_counts = df['trader_id'].value_counts()
        df['trader_id_count'] = df['trader_id'].map(trader_counts)

        # Fill any NaNs
        df.fillna(0, inplace=True)

        # ✅ Features used for prediction (same as training)
        features = [
            'settlement_lag',
            'price_deviation',
            'volume_vs_avg',
            'price_pct_change',
            'volume_pct_change',
            'hour_of_day',
            'rolling_avg_volume',
            'trader_id_count'
        ]

        # Check if all features exist in the uploaded data
        if not all(col in df.columns for col in features):
            st.error("❌ Uploaded CSV is missing some required feature columns.")
        else:
            X_scaled = scaler.transform(df[features])
            df['is_anomaly'] = model.predict(X_scaled)
            df['is_anomaly'] = df['is_anomaly'].map({1: 0, -1: 1})

            # ✅ Show results
            st.subheader("📋 Full Trade Data with Predictions")
            st.dataframe(df)

            st.subheader("🚨 Flagged Anomalies")
            st.dataframe(df[df["is_anomaly"] == 1])

            # ✅ Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions CSV", csv, "anomaly_results.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
