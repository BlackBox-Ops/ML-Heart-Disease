# train.py (versi pipeline lengkap)
import os
import pandas as pd # type: ignore

from sklearn.model_selection import train_test_split # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.preprocessing import StandardScaler, OneHotEncoder # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier # type: ignore
from sklearn.model_selection import GridSearchCV # type: ignore
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix # type: ignore
import joblib # type: ignore

from xgboost import XGBClassifier # type: ignore

# === 1. Load dataset ===
df = pd.read_csv("../data/processed/heart_disease_uci_cleaned.csv")

# Pisahkan fitur & target
X = df.drop(columns=["num"])
y = df["num"]

if "id" in X.columns:
    X = X.drop(columns=["id"])

# Identifikasi kolom numerik & kategorikal
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# === 2. Preprocessing (scaling + one-hot) ===
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

# === 3. Split data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Simpan split data mentah (opsional)
# os.makedirs("../data/processed", exist_ok=True)
# X_train.to_csv("../data/train-data/X_train_raw.csv", index=False)
X_test.to_csv("../data/test-data/X_test_raw.csv", index=False)
# y_train.to_csv("../data/train-data/y_train.csv", index=False)
# y_test.to_csv("../data/test-data/y_test.csv", index=False)

# === 4. Definisikan pipelines lengkap (preprocessing + model) ===
# === 4. Pipelines lengkap ===
pipelines = {
    'SVC': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(probability=True))
    ]),
    'RandomForestClassifier': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ]),
    'GradientBoostingClassifier': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier())
    ]),
    'XGBClassifier': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(eval_metric='logloss', use_label_encoder=False))
    ])
}

# === 5. Parameter grid diperluas ===
param_grids = {
    'SVC': {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__kernel': ['linear', 'rbf', 'poly'],
        'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'classifier__class_weight': [None, 'balanced']
    },
    'RandomForestClassifier': {
        'classifier__n_estimators': [100, 200, 300, 500],
        'classifier__max_depth': [None, 10, 20, 30, 50],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__class_weight': [None, 'balanced']
    },
    'GradientBoostingClassifier': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7],
        'classifier__min_samples_split': [2, 5, 10]
    },
    'XGBClassifier': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7],
        'classifier__subsample': [0.8, 1],
        'classifier__colsample_bytree': [0.8, 1]
    }
}

# === 6. Tuning hyperparameter ===
best_models = {}
for name, pipeline in pipelines.items():
    print(f"\n Mulai tuning untuk {name}...")
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
    print(f"Best CV score for {name}: {grid_search.best_score_:.4f}")

# === 7. Evaluasi & simpan ===
os.makedirs("../models", exist_ok=True)
for name, model in best_models.items():
    print(f"\n=== Evaluasi {name} ===")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Simpan pipeline lengkap
    model_path = os.path.join("../models", f"{name.lower()}_best_pipeline.joblib")
    joblib.dump(model, model_path)
    print(f"Model pipeline lengkap disimpan ke {model_path}")
