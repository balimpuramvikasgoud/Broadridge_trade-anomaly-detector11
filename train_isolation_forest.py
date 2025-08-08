import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ✅ Paths
csv_path = 'C:/Users/Lenovo/OneDrive/Desktop/4-1 P PROJECT/data/synthetic_1500_trade_data.csv'
output_csv = 'C:/Users/Lenovo/OneDrive/Desktop/4-1 P PROJECT/data/trade_anomaly_predictions.csv'
model_path = 'C:/Users/Lenovo/OneDrive/Desktop/4-1 P PROJECT/models/isolation_forest_model.pkl'
scaler_path = 'C:/Users/Lenovo/OneDrive/Desktop/4-1 P PROJECT/models/scaler.pkl'

# ✅ Load data
df = pd.read_csv(csv_path)

# ✅ Simulate additional features
df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='min')
df['hour_of_day'] = df['timestamp'].dt.hour
df['trade_type'] = np.random.choice(['BUY', 'SELL'], size=len(df))
df['rolling_avg_volume'] = df['volume_vs_avg'].rolling(window=10, min_periods=1).mean()
df['trader_id'] = np.random.randint(1000, 1010, size=len(df))
trader_counts = df['trader_id'].value_counts()
df['trader_id_count'] = df['trader_id'].map(trader_counts)

# ✅ Use numeric features only for training
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

# Fill missing values
df.fillna(0, inplace=True)

# Scale features
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
df['is_anomaly'] = model.fit_predict(X_scaled)
df['is_anomaly'] = df['is_anomaly'].map({1: 0, -1: 1})

# Save predictions
df.to_csv(output_csv, index=False)
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print("✅ Model and scaler trained & saved. Predictions stored in data/")
print("✅ Columns in the training dataset:")
print(df.columns.tolist())
