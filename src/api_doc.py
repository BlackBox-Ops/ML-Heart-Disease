import os
import joblib # type: ignore
import pandas as pd # type: ignore
from datetime import datetime

from fastapi import FastAPI, HTTPException # type: ignore
from pydantic import BaseModel, conint, confloat # type: ignore

# ============================================================
# APP CONFIG
# ============================================================
app = FastAPI(
    title="Documentation Heart Disease Prediction API",
    description="""
    API ini digunakan untuk prediksi penyakit jantung berbasis model Machine Learning.
    
    ⚠️ Heart Disease Enterprise Network Notice: 
    - API ini hanya untuk tujuan edukasi & demo.  
    - Tidak menyimpan data pasien.  
    - Hasil prediksi bukan diagnosis medis final  , selalu konsultasikan ke dokter spesialis jantung.
    """,
    version="1.0.0",
    contact={
        "name": "Cardio AI Support",
        "email": "alya.maheswari26@gmail.com"
    },
    license_info={
        "name": "MIT",
    }
)

# ============================================================
# LOAD MODEL
# ============================================================
MODEL_PATH = os.path.join("../models/python-models", "logisticregression_best_pipeline.joblib")

try:
    model = joblib.load(MODEL_PATH)
except Exception:
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")

# ============================================================
# SCHEMA DEFINITIONS
# ============================================================
class Features(BaseModel):
    age: conint(ge=18, le=100)          # type: ignore # Usia (tahun)
    sex: conint(ge=0, le=1)             # type: ignore # Jenis kelamin (0=female,1=male)
    cp: conint(ge=0, le=3)              # type: ignore # Jenis nyeri dada
    trestbps: conint(ge=80, le=200)     # type: ignore # Tekanan darah istirahat
    chol: conint(ge=100, le=600)        # type: ignore # Kolesterol serum
    fbs: conint(ge=0, le=1)             # type: ignore # Gula darah puasa >120 mg/dl
    restecg: conint(ge=0, le=2)         # type: ignore # Hasil EKG
    thalch: conint(ge=60, le=220)       # type: ignore # Detak jantung maksimal
    exang: conint(ge=0, le=1)           # type: ignore # Angina akibat olahraga
    oldpeak: confloat(ge=0, le=6)       # type: ignore # Depresi ST
    slope: conint(ge=0, le=2)           # type: ignore # Slope segmen ST
    ca: conint(ge=0, le=4)              # type: ignore # Jumlah pembuluh besar
    thal: conint(ge=0, le=3)            # type: ignore # Thalassemia

    class Config:
        schema_extra = {
            "example": {
                "age": 55,
                "sex": 1,
                "cp": 0,
                "trestbps": 140,
                "chol": 250,
                "fbs": 0,
                "restecg": 1,
                "thalch": 150,
                "exang": 0,
                "oldpeak": 1.2,
                "slope": 1,
                "ca": 0,
                "thal": 2
            }
        }

class PatientData(BaseModel):
    features: Features

class DiagnosisResult(BaseModel):
    diagnosis: str
    risk_level: str
    probability: dict
    timestamp: str
    notes: str = "Consult with a cardiologist for complete evaluation"

# ============================================================
# LABELS
# ============================================================
DIAGNOSIS_LABELS = {
    0: {"diagnosis": "No Significant Coronary Artery Disease", "risk_level": "Low Risk"},
    1: {"diagnosis": "Coronary Artery Disease Detected", "risk_level": "High Risk"}
}

# ============================================================
# ROUTES
# ============================================================
@app.get("/", tags=["Health Check"])
async def root():
    return {
        "status": "ok",
        "message": "Heart Disease Prediction API is running",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }

@app.post("/diagnose", response_model=DiagnosisResult, tags=["Prediction"])
async def diagnose(patient: PatientData):
    """
    Prediksi penyakit jantung berdasarkan parameter klinis pasien.  
    Hasil berupa **diagnosis awal** + **tingkat risiko** + **probabilitas**.
    """

    try:
        input_data = pd.DataFrame([patient.features.dict()])
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]

        return DiagnosisResult(
            **DIAGNOSIS_LABELS[int(prediction)],
            probability={"healthy": round(float(proba[0]), 4),
                        "disease": round(float(proba[1]), 4)},
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
