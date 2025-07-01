import streamlit as st
import pandas as pd
import joblib
from test_data_processing import process_test_data  # custom processing script

# Load trained features used in the model
with open("models/feature_names.txt", "r") as f:
    feature_names = [line.strip() for line in f.readlines()]

# Load models
lr_model = joblib.load("models/LogisticRegression_model.pkl")
rf_model = joblib.load("models/RandomForest_model.pkl")

# Streamlit UI
st.set_page_config(page_title="Credit Risk Prediction", page_icon="💳")
st.title("💳 Credit Risk Prediction App")
st.markdown("Upload a raw transaction CSV file to predict credit fraud risk.")

# File uploader
uploaded_file = st.file_uploader("📄 Upload raw data file (CSV)", type="csv")

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)

        st.subheader("Raw Input Data")
        st.write(raw_df.head())

        # Process raw data
        processed_df = process_test_data(raw_df)

        # Ensure column order
        processed_df = processed_df[feature_names]

        st.subheader("Processed Features Used for Prediction")
        st.write(processed_df.head())

        # Predictions
        lr_pred = lr_model.predict(processed_df)
        lr_proba = lr_model.predict_proba(processed_df)

        rf_pred = rf_model.predict(processed_df)
        rf_proba = rf_model.predict_proba(processed_df)

        # Display predictions
        st.subheader("🔍 Logistic Regression Predictions")
        st.write(pd.DataFrame({
            'Prediction': ['High Risk' if p == 1 else 'Low Risk' for p in lr_pred],
            'Probability of Fraud': lr_proba[:, 1]
        }))

        st.subheader("🌲 Random Forest Predictions")
        st.write(pd.DataFrame({
            'Prediction': ['High Risk' if p == 1 else 'Low Risk' for p in rf_pred],
            'Probability of Fraud': rf_proba[:, 1]
        }))

    except Exception as e:
        st.error(f"❌ Error while processing or predicting: {e}")

else:
    st.info("Please upload a CSV file with raw transaction data.")
