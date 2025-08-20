from flask import Flask, request, jsonify, render_template, abort # type: ignore
from flask_cors import CORS                                       # type: ignore

import os 
import joblib                                                     # type: ignore
import numpy as np

# -- Konfigurasi --
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), 'models','logisticregression_best_pipeline.joblib'))

# Urutan fitur sesuai sample JSON yang kamu upload
FEATURE_NAMES = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalch","exang","oldpeak","slope","ca","thal"
]

# --- Utils kecil ---
_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileExistsError(f"Model file not found at: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model

def ensure_feature_order(payload: dict):
    """
    Mengembalikan list nilai fitur dalam urutan FEATURE_NAMES.
    - Memaksa nilai numeric (float/int).
    - Melempar ValueError bila fitur hilang atau tidak numeric.
    """
    pass