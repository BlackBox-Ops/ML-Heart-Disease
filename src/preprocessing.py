import pandas as pd  # type: ignore
import numpy as np
import os

# src/data_preprocessing.py
from utils import preprocess_entire_datasheet

# Sklearn Library 
from sklearn.model_selection import train_test_split  # type: ignore

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
categorical_cols = ["sex","cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

# Konversi kolom numerik ke float
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Binarisasi target
df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)

# Hapus baris dengan NaN
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# ---------------------------
# 3. Hapus fitur 'dataset' & 'id' jika ada
# ---------------------------
for col_to_remove in ["dataset", "id"]:
    if col_to_remove in df.columns:
        df.drop(columns=[col_to_remove], inplace=True)
        print(f"Kolom '{col_to_remove}' dihapus dari dataset.")

# Simpan versi cleaned dataset
output_dir = "../data/processed"
os.makedirs(output_dir, exist_ok=True)
cleaned_path = os.path.join(output_dir, "heart_disease_uci_cleaned.csv")
df.to_csv(cleaned_path, index=False)
print(f"Dataset bersih disimpan di {cleaned_path}")

# ---------------------------
# 4. Preprocessing
# ---------------------------
X, y = preprocess_entire_datasheet(df, target_col="num")

# ---------------------------
# 5. Pisahkan X & y
# ---------------------------
X = df.drop(columns=["num"])
y = df["num"]

# Update daftar kolom numerik & kategorikal sesuai dataset final
numerical_cols = [col for col in numerical_cols if col in X.columns]
categorical_cols = [col for col in categorical_cols if col in X.columns]

# ---------------------------
# 6. Split data
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# 7. Save semua hasil
# ---------------------------
X.to_csv(os.path.join(output_dir, "X_processed.csv"), index=False)
y.to_csv(os.path.join(output_dir, "y_processed.csv"), index=False)

X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

print(f"Data split dan disimpan di {output_dir}")