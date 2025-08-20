# app.py
from flask import Flask, request, jsonify, render_template          # type: ignore
from flask_cors import CORS                                         # type: ignore
from pydantic import BaseModel, conint, confloat, ValidationError   # type: ignore
import pandas as pd                                                 # type: ignore
import joblib                                                       # type: ignore
import os

# ============================================================
# CONFIGURATION (Open/Closed Principle: bisa dikembangkan via env var)
# ============================================================
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "models", "logisticregression_best_pipeline.joblib")
)

FEATURE_NAMES = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalch","exang","oldpeak","slope","ca","thal"
]

# ============================================================
# DOMAIN MODEL (Single Responsibility: validasi data input)
# ============================================================
class HeartInput(BaseModel):
    age: conint(ge=18, le=100)              # type: ignore
    sex: conint(ge=0, le=1)                 # type: ignore
    cp: conint(ge=0, le=3)                  # type: ignore
    trestbps: conint(ge=80, le=200)         # type: ignore
    chol: conint(ge=100, le=600)            # type: ignore
    fbs: conint(ge=0, le=1)                 # type: ignore
    restecg: conint(ge=0, le=2)             # type: ignore
    thalch: conint(ge=60, le=220)           # type: ignore
    exang: conint(ge=0, le=1)               # type: ignore
    oldpeak: confloat(ge=0, le=6)           # type: ignore
    slope: conint(ge=0, le=2)               # type: ignore
    ca: conint(ge=0, le=4)                  # type: ignore
    thal: conint(ge=0, le=3)                # type: ignore

# ============================================================
# SERVICE LAYER (Dependency Inversion: controller bergantung ke service)
# ============================================================
class HeartModelService:
    """Service untuk load model & prediksi"""
    def __init__(self, model_path: str):
        self._model_path = model_path
        self._model = None

    def load(self):
        if self._model is None:
            if not os.path.exists(self._model_path):
                raise FileNotFoundError(f"Model file not found: {self._model_path}")
            self._model = joblib.load(self._model_path)
        return self._model

    def predict(self, inputs: list[HeartInput]):
        """Menerima list HeartInput, return prediksi & probabilitas"""
        df = pd.DataFrame([inp.dict() for inp in inputs], columns=FEATURE_NAMES)
        df = df.apply(pd.to_numeric, errors="coerce")

        model = self.load()
        preds = model.predict(df).tolist()
        probas = model.predict_proba(df).tolist() if hasattr(model, "predict_proba") else None
        return preds, probas

# Dependency injection
model_service = HeartModelService(MODEL_PATH)

# ============================================================
# CONTROLLER / FLASK ROUTES (Interface Segregation: pisah endpoint API & UI)
# ============================================================
app = Flask(__name__)
CORS(app)

@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok")

@app.route("/api/metadata", methods=["GET"])
def metadata():
    try:
        model = model_service.load()
        model_name = type(model).__name__
        classes = getattr(model, "classes_", None)
        if classes is not None:
            classes = [int(c) if hasattr(c, "__int__") else c for c in classes]
    except Exception:
        model_name, classes = None, None

    return jsonify(model=model_name, features=FEATURE_NAMES, classes=classes)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify(error="Invalid or missing JSON body"), 400

    records = data if isinstance(data, list) else [data]

    try:
        inputs = [HeartInput(**rec) for rec in records]
    except ValidationError as e:
        return jsonify(error=e.errors()), 400

    try:
        preds, probas = model_service.predict(inputs)
    except Exception as e:
        return jsonify(error=f"Model prediction failed: {e}"), 500

    response = {"predictions": preds}
    if probas is not None:
        response["probabilities"] = probas
    return jsonify(response)

# ---- HTML Form ----
@app.route("/", methods=["GET"])
def form():
    options = {
        "sex": [0, 1],
        "cp": [0, 1, 2, 3],
        "fbs": [0, 1],
        "restecg": [0, 1, 2],
        "exang": [0, 1],
        "slope": [0, 1, 2],
        "ca": [0, 1, 2, 3, 4],
        "thal": [0, 1, 2, 3],
    }
    return render_template("form.html", feature_names=FEATURE_NAMES, options=options)

@app.route("/predict-form", methods=["POST"])
def predict_form():
    form_data = {k: request.form.get(k, None) for k in FEATURE_NAMES}

    try:
        input_obj = HeartInput(**form_data)
    except ValidationError as e:
        return render_template("result.html",
                            error=e.errors(),
                            form_data=form_data,
                            feature_names=FEATURE_NAMES)

    try:
        preds, probas = model_service.predict([input_obj])
    except Exception as e:
        return render_template("result.html",
                            error=f"Model prediction failed: {e}",
                            form_data=form_data,
                            feature_names=FEATURE_NAMES)

    return render_template("result.html",
                        prediction=preds[0],
                        proba=probas[0] if probas else None,
                        form_data=form_data,
                        feature_names=FEATURE_NAMES)

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    debug = bool(int(os.getenv("FLASK_DEBUG", "1")))
    app.run(host=host, port=port, debug=debug)
