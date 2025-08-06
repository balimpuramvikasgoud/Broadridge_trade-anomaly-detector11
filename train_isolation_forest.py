import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ✅ Paths
csv_path = 'C:/Users/Lenovo/OneDrive/Desktop/4-1 P PROJECT/data/large_sample_trade_data.csv'
output_csv = 'C:/Users/Lenovo/OneDrive/Desktop/4-1 P PROJECT/data/trade_anomaly_predictions.csv'
model_path = 'C:/Users/Lenovo/OneDrive/Desktop/4-1 P PROJECT/models/isolation_forest_model.pkl'
scaler_path = 'C:/Users/Lenovo/OneDrive/Desktop/4-1 P PROJECT/models/scaler.pkl'

# Load data
df = pd.read_csv(csv_path)

# Select features
features = ['settlement_lag', 'price_deviation', 'volume_vs_avg']
X = df[features]

# ✅ Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
df['is_anomaly'] = model.fit_predict(X_scaled)
df['is_anomaly'] = df['is_anomaly'].map({1: 0, -1: 1})  # 0 = normal, 1 = anomaly

# Save predictions
df.to_csv(output_csv, index=False)

# Save model and scaler
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print("✅ Model and scaler trained & saved. Predictions stored in data/")
