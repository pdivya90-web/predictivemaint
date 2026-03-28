
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# Define a Pydantic model for input data validation
class PredictionInput(BaseModel):
    engine_rpm: float
    lub_oil_pressure: float
    fuel_pressure: float
    coolant_pressure: float
    lub_oil_temp: float
    coolant_temp: float

app = FastAPI()

# Load the model
MODEL_PATH = "best_xgboost_model.joblib" # This path is relative to the WORKDIR in Dockerfile
loaded_model = joblib.load(MODEL_PATH)

@app.get("/")
async def read_root():
    return {"message": "XGBoost Predictive Maintenance Model API"}

@app.post("/predict")
async def predict(data: PredictionInput):
    # Convert input data to pandas DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Make prediction
    prediction = loaded_model.predict(input_df)[0]
    probability = loaded_model.predict_proba(input_df)[0].tolist()

    # Map prediction to human-readable format
    result = "Failing" if prediction == 1 else "Healthy"

    return {
        "prediction": result,
        "probability_healthy": probability[0],
        "probability_failing": probability[1]
    }
