import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = "C:/Users/Lenovo/OneDrive/Desktop/4-1 P PROJECT/models/isolation_forest_model.pkl"
SCALER_PATH = "C:/Users/Lenovo/OneDrive/Desktop/4-1 P PROJECT/models/scaler.pkl"

# Load model & scaler
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    st.error("❌ Model or scaler file not found.")
    st.stop()

uploaded_file = st.file_uploader("Upload your trade CSV file (raw data)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    features = ['settlement_lag', 'price_deviation', 'volume_vs_avg']

    if all(col in df.columns for col in features):
        X_scaled = scaler.transform(df[features])
        predictions = model.predict(X_scaled)
        df['is_anomaly'] = pd.Series(predictions).map({1: 0, -1: 1})

        st.subheader("📊 Full Trade Data with Predictions")
        st.dataframe(df)

        anomalies = df[df["is_anomaly"] == 1]
        st.subheader("🚨 Flagged Anomalies")
        st.dataframe(anomalies)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions CSV", csv, "anomaly_results.csv", "text/csv")
    else:
        st.error(f"❌ Uploaded CSV must contain columns: {features}")
