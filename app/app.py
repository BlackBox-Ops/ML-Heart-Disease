from flask import Flask, request, render_template   # type: ignore
from pydantic import BaseModel, conint, confloat    # type: ignore
import pandas as pd                                 # type: ignore
import joblib                                       # type: ignore
import os
from datetime import datetime

# ============================================================
# KONFIG
# ============================================================
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models", "logisticregression_best_pipeline.joblib"
)

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalch", "exang", "oldpeak", "slope", "ca", "thal"
]

# ============================================================
# VALIDASI INPUT
# ============================================================
class HeartInput(BaseModel):
    age: conint(ge=18, le=100)                  # type: ignore
    sex: conint(ge=0, le=1)                     # type: ignore
    cp: conint(ge=0, le=3)                      # type: ignore
    trestbps: conint(ge=80, le=200)             # type: ignore
    chol: conint(ge=100, le=600)                # type: ignore
    fbs: conint(ge=0, le=1)                     # type: ignore
    restecg: conint(ge=0, le=2)                 # type: ignore
    thalch: conint(ge=60, le=220)               # type: ignore
    exang: conint(ge=0, le=1)                   # type: ignore
    oldpeak: confloat(ge=0, le=6)               # type: ignore
    slope: conint(ge=0, le=2)                   # type: ignore
    ca: conint(ge=0, le=4)                      # type: ignore
    thal: conint(ge=0, le=3)                    # type: ignore

# ============================================================
# LOAD MODEL
# ============================================================
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

model = load_model()

DISCLAIMER = "⚠️ This demo does not store patient data. Prediction results are not a medical diagnosis."

# ============================================================
# DISPLAY MAP UNTUK HASIL
# ============================================================
DISPLAY_MAP = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"},
    "fbs": {0: "≤120 mg/dl", 1: ">120 mg/dl"},
    "restecg": {0: "Normal", 1: "ST-T abnormality", 2: "LV Hypertrophy"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {0: "Upsloping", 1: "Flat", 2: "Downsloping"},
    "ca": {0: "0", 1: "1", 2: "2", 3: "3", 4: "4"},
    "thal": {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect", 3: "Unknown"}
}

# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("form.html", disclaimer=DISCLAIMER)

@app.route("/predict-form", methods=["POST"])
def predict():
    try:
        form_data = request.form.to_dict()

        # langsung cast ke int/float
        input_dict = {
            "age": int(form_data["age"]),
            "sex": int(form_data["sex"]),
            "cp": int(form_data["cp"]),
            "trestbps": int(form_data["trestbps"]),
            "chol": int(form_data["chol"]),
            "fbs": int(form_data["fbs"]),
            "restecg": int(form_data["restecg"]),
            "thalch": int(form_data["thalch"]),
            "exang": int(form_data["exang"]),
            "oldpeak": float(form_data["oldpeak"]),
            "slope": int(form_data["slope"]),
            "ca": int(form_data["ca"]),
            "thal": int(form_data["thal"]),
        }

        # Validasi
        HeartInput(**input_dict)

        # Prediksi
        df = pd.DataFrame([input_dict], columns=FEATURE_NAMES)
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0] if hasattr(model, "predict_proba") else None

        # buat display-friendly
        form_display = {}
        for k, v in input_dict.items():
            if k in DISPLAY_MAP:
                form_display[k] = DISPLAY_MAP[k].get(v, v)
            else:
                form_display[k] = v

        return render_template(
            "result.html",
            prediction=int(pred),
            probability=prob.tolist() if prob is not None else None,
            disclaimer=DISCLAIMER,
            form_data=form_display,  # kirim yg sudah readable
            feature_names=FEATURE_NAMES,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

    except Exception as e:
        return render_template(
            "result.html",
            error=str(e),
            disclaimer=DISCLAIMER,
            form_data=request.form.to_dict(),
            feature_names=FEATURE_NAMES,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
