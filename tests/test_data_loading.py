import os 
import pandas as pd #type: ignore
import pytest #type: ignore


def test_raw_exits():
    assert os.path.exists("../data/processed/heart_disease_uci_cleaned.csv"), "Datasheet tidak ditemukan"

def test_raw_data_schema():
    df = pd.read_csv("../data/processed/heart_disease_uci_cleaned.csv")
    expected_cols = {"age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalch", "exang", "oldpeak", "slope", "ca", "thal", "num"}
    assert expected_cols.issubset(df.columns),"Schema columns tidak sesuai"

def test_target_no_missing():
    df = pd.read_csv("../data/processed/heart_disease_uci_cleaned.csv")
    assert df["num"].notna().all(), "Target memiliki nilai NaN"
