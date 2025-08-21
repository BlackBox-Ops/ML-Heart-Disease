import joblib # type: ignore
import pandas as pd # type: ignore
import json
import os

model = joblib.load(os.path.join("../models/python-models", "logisticregression_best_pipeline.joblib"))

with open("../data/raw/sample_test_data.json", "r") as f:
    df_new = pd.DataFrame(json.load(f))

# Jika ada 'num', buang
if "num" in df_new.columns:
    df_new = df_new.drop(columns=["num"])

import numpy as np

def align_to_preprocessor_columns(model, df):
    """
    Pastikan df punya kolom persis seperti yang dipakai preprocessor saat fit.
    Kolom yang hilang akan ditambah dengan NaN (nanti di-impute oleh pipeline).
    Urutan kolom juga disamakan.
    """
    # Ambil step preprocessor dari pipeline
    pre = model.named_steps.get('preprocessor', None)
    if pre is None:
        raise ValueError("Model tidak punya step 'preprocessor'. Pastikan model adalah pipeline lengkap.")

    expected_cols = []
    for name, trans, cols in pre.transformers_:
        # cols bisa berupa list nama kolom
        if isinstance(cols, (list, tuple)):
            expected_cols.extend(list(cols))

    # Tambahkan kolom yang hilang sebagai NaN
    missing = [c for c in expected_cols if c not in df.columns]
    for c in missing:
        df[c] = np.nan

    # **Opsional**: buang kolom ekstra yang tidak dipakai preprocessor
    df = df[expected_cols]

    return df


# 👉 Samakan kolom dengan yang diharapkan preprocessor (memperbaiki error 'columns are missing: {id}')
df_new = align_to_preprocessor_columns(model, df_new)

pred = model.predict(df_new)
prob = model.predict_proba(df_new)[:, 1] if hasattr(model, "predict_proba") else None

out = df_new.copy()
out["prediction"] = pred
if prob is not None:
    out["probability"] = prob

print(out)
# out.to_csv("../data/processed/new_data_predictions.csv", index=False)
