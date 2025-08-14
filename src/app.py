import os
import joblib # type: ignore
from fastapi import FastAPI, HTTPException # type: ignore
from pydantic import BaseModel # type: ignore
import pandas as pd # type: ignore

app = FastAPI(title="Heart Disease Prediction API")

# Path model
MODEL_PATH = os.path.join("../models", "svc_best_pipeline.joblib")
model = joblib.load(MODEL_PATH)

# Required feature columns
FEATURE_COLUMNS = [
    'age', 'sex', 'dataset', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# Model input
class PatientData(BaseModel):
    features: dict

# Model output
class DiagnosisResult(BaseModel):
    diagnosis: str
    risk_level: str
    notes: str = "Consult with a cardiologist for complete evaluation"

# Medical diagnosis labels
DIAGNOSIS_LABELS = {
    0: {
        "diagnosis": "No Significant Coronary Artery Disease",
        "risk_level": "Low Risk"
    },
    1: {
        "diagnosis": "Coronary Artery Disease Detected",
        "risk_level": "High Risk"
    }
}

@app.post("/diagnose", response_model=DiagnosisResult)
async def diagnose(patient: PatientData):
    # Validate input features
    missing = [col for col in FEATURE_COLUMNS if col not in patient.features]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required clinical parameters: {', '.join(missing)}"
        )

    # Prepare input data frame
    input_data = pd.DataFrame([patient.features], columns=FEATURE_COLUMNS)

    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Return intuitive medical diagnosis
    return DiagnosisResult(
        **DIAGNOSIS_LABELS[int(prediction)]
    )

# Cara menjalankan uvicorn app:app --reload

