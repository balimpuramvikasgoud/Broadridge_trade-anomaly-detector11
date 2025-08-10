import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Streamlit Page Config ---
st.set_page_config(page_title="Trade Anomaly Detector", layout="wide")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        body {
            background-color: #f4f6f8;
        }
        .main-title {
            font-size: 36px;
            color: #2E86C1;
            font-weight: bold;
        }
        .highlight-anomaly {
            background-color: #ffcccc;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown('<p class="main-title">📈 Trade Anomaly Detection & Explanation</p>', unsafe_allow_html=True)

# --- Paths to Model and Scaler ---
MODEL_PATH = "C:/Users/Lenovo/OneDrive/Desktop/4-1 P PROJECT/models/isolation_forest_model.pkl"
SCALER_PATH = "C:/Users/Lenovo/OneDrive/Desktop/4-1 P PROJECT/models/scaler.pkl"

# --- Load Model and Scaler ---
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("❌ Model or scaler file not found. Please train the model first.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# --- File Upload ---
uploaded_file = st.file_uploader("📤 Upload Trade CSV File", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # --- Feature Engineering (Must Match Training) ---
        df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='min')
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['trade_type'] = np.random.choice(['BUY', 'SELL'], size=len(df))
        df['rolling_avg_volume'] = df['volume_vs_avg'].rolling(window=10, min_periods=1).mean()
        df['trader_id'] = np.random.randint(1000, 1010, size=len(df))
        trader_counts = df['trader_id'].value_counts()
        df['trader_id_count'] = df['trader_id'].map(trader_counts)

        # Fill missing values
        df.fillna(0, inplace=True)

        # --- Features Used for Prediction ---
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

        if not all(col in df.columns for col in features):
            st.error("❌ Uploaded CSV is missing required columns.")
        else:
            X_scaled = scaler.transform(df[features])
            df['is_anomaly'] = model.predict(X_scaled)
            df['is_anomaly'] = df['is_anomaly'].map({1: 0, -1: 1})

            # --- Highlight Function ---
            def highlight_anomalies(row):
                return ['background-color: #ffcccc' if row['is_anomaly'] == 1 else '' for _ in row]

            # --- Display Full Data ---
            st.subheader("📋 Full Trade Data with Predictions")
            st.dataframe(df.style.apply(highlight_anomalies, axis=1))

            # --- Display Only Anomalies ---
            st.subheader("🚨 Flagged Anomalies")
            st.dataframe(df[df["is_anomaly"] == 1])

            # --- Download Predictions ---
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions CSV", csv, "anomaly_results.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
