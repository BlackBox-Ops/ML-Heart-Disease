import pandas as pd # type: ignore
import numpy as np
import os

# src/data_preprocessing.py
from utils import preprocess_entire_datasheet

# Sklearn Library 
from sklearn.model_selection import train_test_split # type: ignore

# ---------------------------
# 1. Load Dataset
# ---------------------------
data_path = "../data/raw/heart_disease_uci.csv"

if not os.path.exists(data_path):
    print(f"Error: File '{data_path}' tidak ditemukan.")
    exit()

df = pd.read_csv(data_path)

# ---------------------------
# 2. Cleaning Data
# ---------------------------

# Ganti '?' dengan NaN
df.replace("?", np.nan, inplace=True)

# Kolom numerik & kategorikal
numerical_cols = ["age", "trestbps", "chol", "thalch", "oldpeak"]
categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

# Konversi kolom numerik ke float
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Binarisasi target
df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)

# Hapus baris dengan NaN
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# ---------------------------
# 3. Preprocessing
# ---------------------------
X, y = preprocess_entire_datasheet(df, target_col="num")

# ---------------------------
# 4. Pisahkan X & y
# ---------------------------
drop_cols = ["num", "dataset","id"]
X = df.drop(columns=[col for col in drop_cols if col in df.columns])
y = df["num"]


# Update daftar kolom
numerical_cols = [col for col in numerical_cols if col in X.columns]
categorical_cols = [col for col in categorical_cols if col in X.columns]

# ---------------------------
# 5. Split data
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    )

output_dir="../data/processed"
output_test_data="../data/test-data"
output_train_data="../data/train-data"

"""Simpan X dan y ke folder output_dir"""
os.makedirs(output_dir, exist_ok=True)  # Buat folder jika belum ada

X_path = os.path.join(output_dir, "X_processed.csv")
y_path = os.path.join(output_dir, "y_processed.csv")

X.to_csv(X_path, index=False)
y.to_csv(y_path, index=False)

# Simpan masing-masing
X_train.to_csv(os.path.join(output_train_data, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(output_test_data, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(output_train_data, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(output_test_data, "y_test.csv"), index=False)

print(f"Data split dan disimpan di {output_dir}, {output_train_data}, {output_test_data}")



