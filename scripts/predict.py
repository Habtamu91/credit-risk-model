
import numpy as np
import pandas as pd
import joblib
import os

# Define path to model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "gradient_boosting.pkl")

# Load trained model
model = joblib.load(MODEL_PATH)


def predict_single(input_features: list):
    """
    input_features: A list in the order:
        [Total_Amount, Avg_Amount, Std_Amount, Num_Transactions,
         Unique_Hours, Transaction_Hour, Transaction_Day,
         Transaction_Month, Transaction_Year]
    Returns: risk probability and high-risk flag
    """
    input_array = np.array([input_features])
    risk_prob = model.predict_proba(input_array)[0][1]
    is_high_risk = int(risk_prob > 0.5)
    return is_high_risk, round(risk_prob, 4)


# Example usage
if __name__ == "__main__":
    # Example customer features
    sample_input = [
        2500.0,     # Total_Amount
        500.0,      # Avg_Amount
        300.0,      # Std_Amount
        5.0,        # Num_Transactions
        3.0,        # Unique_Hours
        14.0,       # Transaction_Hour
        15.0,       # Transaction_Day
        6.0,        # Transaction_Month
        2025.0      # Transaction_Year
    ]

    is_high_risk, prob = predict_single(sample_input)
    print("Risk Probability:", prob)
    print("Predicted is_high_risk:", is_high_risk)
