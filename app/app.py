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
