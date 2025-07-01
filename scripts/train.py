import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# ======================
# Load Processed Dataset
# ======================
data_path = "data/processed/processed_data.csv"
df = pd.read_csv(data_path)

# Drop unnecessary identifier and timestamp columns if present
drop_cols = [
    "TransactionId", "BatchId", "SubscriptionId",
    "CustomerId", "TransactionStartTime"
]
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

# Separate features and target
X = df.drop("is_high_risk", axis=1)
y = df["is_high_risk"]

# Drop non-numeric features
X = X.select_dtypes(include=[np.number])

# ✅ Save feature names for use in predict.py
feature_names = list(X.columns)
os.makedirs("models", exist_ok=True)
with open("models/feature_names.txt", "w") as f:
    for name in feature_names:
        f.write(f"{name}\n")

# ==========================
# Train/Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ==========================
# Define Models
# ==========================
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(random_state=42, class_weight="balanced")
}

# Correct pipeline step name for GridSearch
param_grid_rf = {
    "RandomForest__n_estimators": [100, 200],
    "RandomForest__max_depth": [None, 10, 20]
}

# ==========================
# MLflow Logging
# ==========================
mlflow.set_experiment("credit-risk-model")

for model_name, model in models.items():
    print(f"\n🔹 Training {model_name}...")

    with mlflow.start_run(nested=True):
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            (model_name, model)
        ])

        if model_name == "RandomForest":
            grid = GridSearchCV(
                pipeline,
                param_grid_rf,
                cv=3,
                scoring="roc_auc",
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
        else:
            best_model = pipeline.fit(X_train, y_train)

        # Predictions
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        print(f"{model_name} ROC-AUC: {roc_auc:.4f}")

        # Log to MLflow
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Save model locally
        model_path = f"models/{model_name}_model.pkl"
        joblib.dump(best_model, model_path)

        # Log model to MLflow
        mlflow.sklearn.log_model(best_model, model_name)

        # Optionally register best model
        if model_name == "RandomForest":
            mlflow.sklearn.log_model(
                best_model,
                artifact_path="rf_model",
                registered_model_name="CreditRiskRandomForest"
            )
