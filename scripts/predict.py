import joblib
import numpy as np
import pandas as pd

# Load trained model
model_path = "models/RandomForest_model.pkl"
model = joblib.load(model_path)

# Load the saved feature names
with open("models/feature_names.txt", "r") as f:
    feature_names = [line.strip() for line in f]

# Example input in correct order — UPDATE values with real test case
sample_values = [
    50.0,    # Amount
    75.0,    # Avg_Amount
    256,     # CountryCode
    0,       # FraudResult
    5,       # Num_Transactions
    3,       # Num_Channels
    1,       # Month
    5,       # Weekday
    1,       # IsWeekend
    10.0,    # Recency
    4.0,     # Frequency
    300.0,   # Monetary
    1,       # PricingStrategy
    0        # ChannelId_xxx (if one-hot encoded)
]

# Create DataFrame with correct column names
sample_df = pd.DataFrame([sample_values], columns=feature_names)

# Predict
pred_class = model.predict(sample_df)[0]
pred_prob = model.predict_proba(sample_df)[0][1]

print(f"Predicted Class (is_high_risk): {pred_class}")
print(f"Predicted Risk Probability: {round(pred_prob, 4)}")
