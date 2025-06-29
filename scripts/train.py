
import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


df = pd.read_csv("../data/raw/data.csv")


def get_features_and_target(df):
    # Drop non-numeric or irrelevant columns
    drop_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'TransactionStartTime']
    df = df.drop(columns=drop_cols, errors='ignore')

    X = df.drop(columns=['is_high_risk'])
    y = df['is_high_risk']
    return X, y


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }


def train_and_log_model(model, model_name, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        mlflow.sklearn.log_model(model, model_name)
        print(f"✅ {model_name} logged to MLflow")

        return model


if __name__ == "__main__":
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_data.csv")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load and prepare data
    df = pd.read_csv(PROCESSED_PATH)
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Start MLflow tracking
    mlflow.set_experiment("Credit Risk Modeling")

    # Train and log Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    trained_lr = train_and_log_model(lr_model, "Logistic_Regression", X_train, y_train, X_test, y_test)
    joblib.dump(trained_lr, os.path.join(MODEL_DIR, "logistic_regression.pkl"))

    # Train and log Gradient Boosting
    gb_model = GradientBoostingClassifier()
    trained_gb = train_and_log_model(gb_model, "Gradient_Boosting", X_train, y_train, X_test, y_test)
    joblib.dump(trained_gb, os.path.join(MODEL_DIR, "gradient_boosting.pkl"))
