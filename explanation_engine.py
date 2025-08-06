import pandas as pd

# Load predictions file
df = pd.read_csv('C:/Users/Lenovo/OneDrive/Desktop/4-1 P PROJECT/data/trade_anomaly_predictions.csv')

# Function to explain anomalies
def generate_explanation(row):
    reasons = []
    if row['settlement_lag'] > 3:
        reasons.append(f"Late settlement by {row['settlement_lag']} days")
    if abs(row['price_deviation']) > 10:
        reasons.append(f"Price deviation of {row['price_deviation']:.2f}%")
    if row['volume_vs_avg'] > 1.5:
        reasons.append(f"High volume: {row['volume_vs_avg']:.2f}x avg")
    return " | ".join(reasons) if reasons else "No clear reason"

# Apply to anomalies only
df['explanation'] = df.apply(lambda row: generate_explanation(row) if row['is_anomaly'] == 1 else '', axis=1)

# Save final CSV
df.to_csv('C:/Users/Lenovo/OneDrive/Desktop/4-1 P PROJECT/data/trade_anomaly_with_explanations.csv', index=False)

print("✅ Explanations generated and saved to data/trade_anomaly_with_explanations.csv")
