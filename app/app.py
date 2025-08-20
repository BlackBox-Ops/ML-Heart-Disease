# app.py
from flask import Flask, request, jsonify, render_template  # type: ignore
from flask_cors import CORS                                 # type: ignore
import os
import joblib                                               # type: ignore 
import pandas as pd                                         # type: ignore 

# --- Konfigurasi ---
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "models", "logisticregression_best_pipeline.joblib")
)

# Urutan fitur sesuai sample JSON
FEATURE_NAMES = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalch","exang","oldpeak","slope","ca","thal"
]

# --- Utils kecil ---
_model = None

def get_model():
    """Load model hanya sekali."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model

def ensure_feature_order(payload: dict):
    """
    Mengembalikan list nilai fitur dalam urutan FEATURE_NAMES.
    - Memaksa numeric (float/int).
    - Raise ValueError bila ada fitur hilang/tidak numeric.
    """
    ordered = []
    for key in FEATURE_NAMES:
        if key not in payload:
            raise ValueError(f"Missing required feature: {key}")
        val = payload[key]
        try:
            if isinstance(val, str) and val.strip() == "":
                raise ValueError(f"Empty value for feature: {key}")
            f = float(val)
            if f.is_integer():
                ordered.append(int(f))
            else:
                ordered.append(f)
        except Exception:
            raise ValueError(f"Invalid value for feature '{key}': {val} (must be numeric)")
    return ordered

# --- Flask app ---
app = Flask(__name__)
CORS(app)

@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok")

@app.route("/api/metadata", methods=["GET"])
def metadata():
    try:
        model = get_model()
        model_name = type(model).__name__
        classes = getattr(model, "classes_", None)
        if classes is not None:
            try:
                classes = [int(c) if hasattr(c, "__int__") else c for c in classes]
            except Exception:
                pass
    except Exception:
        model_name = None
        classes = None

    return jsonify(model=model_name, features=FEATURE_NAMES, classes=classes)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        model = get_model()
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 500

    data = request.get_json(silent=True)
    if data is None:
        return jsonify(error="Invalid or missing JSON body"), 400

    records = data if isinstance(data, list) else [data]

    try:
        rows = [ensure_feature_order(r) for r in records]
    except ValueError as e:
        return jsonify(error=str(e)), 400

    # Pandas DataFrame sesuai FEATURE_NAMES
    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    X = X.apply(pd.to_numeric, errors="coerce")

    try:
        preds = model.predict(X).tolist()
    except Exception as e:
        return jsonify(error=f"Model prediction failed: {e}"), 500

    response = {"predictions": preds}
    if hasattr(model, "predict_proba"):
        try:
            response["probabilities"] = model.predict_proba(X).tolist()
        except Exception:
            pass

    return jsonify(response)

# ---- HTML form ----
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
    try:
        model = get_model()
    except FileNotFoundError as e:
        return render_template("result.html", error=str(e), form_data={})

    form_data = {k: request.form.get(k, "") for k in FEATURE_NAMES}

    try:
        row = ensure_feature_order(form_data)
    except ValueError as e:
        return render_template("result.html", error=str(e),
                            form_data=form_data, feature_names=FEATURE_NAMES)

    X = pd.DataFrame([row], columns=FEATURE_NAMES)
    X = X.apply(pd.to_numeric, errors="coerce")

    try:
        pred = model.predict(X).tolist()[0]
    except Exception as e:
        return render_template("result.html",
                            error=f"Model prediction failed: {e}",
                            form_data=form_data, feature_names=FEATURE_NAMES)

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X).tolist()[0]
        except Exception:
            pass

    return render_template("result.html",
                        prediction=pred, proba=proba,
                        form_data=form_data, feature_names=FEATURE_NAMES)

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    debug = bool(int(os.getenv("FLASK_DEBUG", "1")))
    app.run(host=host, port=port, debug=debug)
