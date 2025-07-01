# scripts/api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc

app = FastAPI()

# Load production model from MLflow
model = mlflow.pyfunc.load_model("models:/CreditRiskModel/Production")

class CustomerData(BaseModel):
    feature1: float
    feature2: float
    # ... all 14 features

@app.post("/predict")
def predict(data: CustomerData):
    prediction = model.predict([data.dict().values()])[0]
    proba = model.predict_proba([data.dict().values()])[0][1]
    return {
        "risk_class": int(prediction),
        "risk_probability": float(proba),
        "risk_category": "high" if prediction == 1 else "low"
    }