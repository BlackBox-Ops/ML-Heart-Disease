import pandas as pd # type: ignore
import sys, os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import handle_missing_values, winsorize_data, preprocess_features

def test_handle_missing_values():
    df = pd.DataFrame({"num_col": [1, None, 3], "cat_col": ["A", None, "B"]})
    df_filled = handle_missing_values(df)
    assert df_filled.isna().sum().sum() == 0, "Masih ada NaN setelah imputasi"

def test_winsorize_data():
    df = pd.DataFrame({"num_col": [1, 2, 100]})
    df_win = winsorize_data(df)
    assert df_win["num_col"].max() <= 100, "Winsorize gagal"

def test_preprocess_features_shape():
    df = pd.DataFrame({
        "num_col": [1, 2, 3],
        "cat_col": ["A", "B", "A"]
    })
    df_proc = preprocess_features(df)
    assert df_proc.shape[0] == 3, "Jumlah baris berubah"
    assert df_proc.shape[1] > 1, "Tidak ada fitur hasil encoding"