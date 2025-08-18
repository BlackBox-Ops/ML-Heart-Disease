# train.py
import os
import joblib # type: ignore
import numpy as np
import pandas as pd # type: ignore

from sklearn.model_selection import train_test_split, GridSearchCV # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.preprocessing import StandardScaler, OneHotEncoder # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import classification_report, accuracy_score # type: ignore

from skl2onnx import convert_sklearn  # type: ignore
from skl2onnx.common.data_types import FloatTensorType  # type: ignore

# ============================================================
# 1. Load Dataset
# ============================================================
data = pd.read_csv("../data/processed/heart_disease_uci_cleaned.csv")

# Pisahkan fitur & target
X = data.drop(columns=["num"])
y = data["num"]

if "id" in X.columns:
    X = X.drop(columns=["id"])


# Pisahkan kolom numerik & kategorikal
num_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

# ============================================================
# 2. Preprocessing Pipeline
# ============================================================
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ]
)

# ============================================================
# 3. Model Candidates
# ============================================================
models = {
    "LogisticRegression": LogisticRegression(max_iter=500),
    "RandomForest": RandomForestClassifier(),
    "SVC": SVC(probability=True)
}

params = {
    "LogisticRegression": {"classifier__C": [0.1, 1, 10]},
    "RandomForest": {"classifier__n_estimators": [100, 200]},
    "SVC": {"classifier__C": [0.1, 1, 10], "classifier__kernel": ["linear", "rbf"]}
}

# ============================================================
# 4. Train & Evaluate
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

best_models = {}

for name, clf in models.items():
    print(f"\n🔹 Training {name} ...")
    pipe = Pipeline(steps=[("preprocessor", preprocessor),
                        ("classifier", clf)])
    grid = GridSearchCV(pipe, params[name], cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Best params: {grid.best_params_}")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    best_models[name] = grid.best_estimator_

# ============================================================
# 5. Save Best Pipeline (Joblib)
# ============================================================
os.makedirs("../models", exist_ok=True)

for name, model in best_models.items():
    joblib_path = os.path.join("../models/python-models", f"{name.lower()}_best_pipeline.joblib")
    joblib.dump(model, joblib_path)
    print(f"Pipeline {name} disimpan ke {joblib_path}")

# ============================================================
# 6. Save ONNX (Estimator-only untuk Java)
# ============================================================
if "LogisticRegression" in best_models:
    svc_pipeline = best_models["LogisticRegression"]
    svc_only = svc_pipeline.named_steps["classifier"]

    try:
        n_features = len(num_cols) + len(cat_cols)
        initial_types = [("float_input", FloatTensorType([None, n_features]))]

        onnx_model = convert_sklearn(svc_only, initial_types=initial_types)
        onnx_path = os.path.join("../models/java-models", "Logreg_only.onnx")

        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        print(f"Estimator SVC-only berhasil disimpan ke {onnx_path}")
    except Exception as e:
        print("Gagal convert estimator ke ONNX:", e)
