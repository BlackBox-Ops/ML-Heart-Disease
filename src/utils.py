import pandas as pd    # type: ignore
import numpy as np     # type: ignore
import joblib          # type: ignore 
import os       

# Preprocessing data library for sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # type: ignore
from sklearn.compose import ColumnTransformer                    # type: ignore
from sklearn.pipeline import Pipeline                            # type: ignore
from sklearn.impute import SimpleImputer                         # type: ignore
from sklearn.model_selection import GridSearchCV                 # type: ignore

# Import winsorize (kompatibel untuk SciPy baru & lama)
from scipy.stats.mstats import winsorize

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.metrics import ( #type: ignore 
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Isi nilai kosong: mean untuk numerik, mode untuk kategorikal."""
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in num_cols:
        df[col].fillna(df[col].mean(), inplace=True)

    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df


def winsorize_data(df: pd.DataFrame, limits: tuple = (0.05, 0.05)) -> pd.DataFrame:
    """Winsorize kolom numerik untuk mengurangi efek outlier."""
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in num_cols:
        df[col] = winsorize(df[col], limits=limits)
    return df


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scaling numerik & one-hot encoding kategorikal."""
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )

    df_processed = preprocessor.fit_transform(df)

    # Ambil nama kolom hasil transformasi
    feature_names = (
        list(num_cols) +
        list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols))
    )

    return pd.DataFrame(df_processed, columns=feature_names)


def preprocess_entire_datasheet(df: pd.DataFrame, target_col: str = "num"):
    """
    Full preprocessing:
    1. Pisahkan target
    2. Handle missing values
    3. Winsorize
    4. Scaling + One-hot encoding
    """
    if target_col not in df.columns:
        raise KeyError(f"Kolom target '{target_col}' tidak ditemukan.")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    X = handle_missing_values(X)
    X = winsorize_data(X)
    X = preprocess_features(X)

    return X, y


def tune_models(pipelines, param_grids, X_train, y_train):
    """
    Lakukan hyperparameter tuning untuk semua pipeline model.
    Return dictionary best_models.
    """
    best_models = {}
    for name, pipeline in pipelines.items():
        print(f"\nMulai tuning untuk {name}...")
        grid_search = GridSearchCV(
            pipeline,
            param_grids[name],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Best cross-validation accuracy for {name}: {grid_search.best_score_:.4f}")
    return best_models


def evaluate_model(model, X_test, y_test):
    """
    Cetak classification report & confusion matrix untuk model.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return acc


def save_model(model, path):
    """
    Simpan model (pipeline lengkap) ke path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model disimpan ke {path}")
