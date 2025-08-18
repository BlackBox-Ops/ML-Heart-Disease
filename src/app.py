import os               #type: ignore
import joblib           #type: ignore 
import pandas as pd     #type: ignore

from fastapi import FastAPI, HTTPException          #type: ignore
from pydantic import BaseModel, conint, confloat    #type: ignore 

app = FastAPI(title="Heart Disease Prediction API")

# Path model 
MODEL_PATH = os.path.join("../models","svc_best_pipeline.joblib")
model = joblib.load(MODEL_PATH)

# model input dengan validasi medis 
class Features(BaseModel) :
    age: conint(ge=18, le=100)          # type: ignore # usia realistis
    sex: conint(ge=0, le=1)             # type: ignore # 0 = female, 1 = male
    cp: conint(ge=0, le=3)              # type: ignore # chest pain type
    trestbps: conint(ge=80, le=200)     # type: ignore # resting blood pressure
    chol: conint(ge=100, le=600)        # type: ignore # serum cholestoral
    fbs: conint(ge=0, le=1)             # type: ignore # fasting blood sugar
    restecg: conint(ge=0, le=2)         # type: ignore # resting ECG results
    thalch: conint(ge=60, le=220)       # type: ignore # max heart rate achieved
    exang: conint(ge=0, le=1)           # type: ignore # exercise induced angina
    oldpeak: confloat(ge=0, le=6)       # type: ignore # ST depression
    slope: conint(ge=0, le=2)           # type: ignore # slope of peak exercise ST
    ca: conint(ge=0, le=4)              # type: ignore # number of major vessels
    thal: conint(ge=0, le=3)            # type: ignore # thalassemia

class PatientData(BaseModel):
    features: Features

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
    # Convert pydantic model ke dataframe
    input_data = pd.DataFrame([patient.features.dict()])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Return intuitive medical diagnosis
    return DiagnosisResult(
        **DIAGNOSIS_LABELS[int(prediction)]
    )

# Run server:
# uvicorn app:app --reload