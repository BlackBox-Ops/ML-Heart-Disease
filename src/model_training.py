# src/model_training.py
import pandas as pd # type: ignore
from sklearn.model_selection import GridSearchCV # type: ignore
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix # type: ignore
import joblib # type: ignore


def tune_models(pipelines, param_grids, X_train, y_train, cv=5, scoring='accuracy'):
    """
    Lakukan GridSearchCV untuk semua pipeline di 'pipelines' menggunakan parameter di 'param_grids'.
    
    Parameters:
    - pipelines: dict {model_name: pipeline_object}
    - param_grids: dict {model_name: parameter_grid}
    - X_train, y_train: data latih
    - cv: jumlah cross-validation folds
    - scoring: metrik evaluasi
    
    Returns:
    - best_models: dict {model_name: best_pipeline}
    """
    best_models = {}
    for name, pipeline in pipelines.items():
        print(f"\n Mulai tuning untuk {name}...")
        grid_search = GridSearchCV(
            pipeline,
            param_grids[name],
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Best cross-validation {scoring} for {name}: {grid_search.best_score_:.4f}")
    return best_models


def evaluate_model(model, X_test, y_test):
    """
    Evaluasi model dengan data test.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", round(acc, 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return acc


def save_model(model, filename):
    """
    Simpan model ke file .joblib
    """
    joblib.dump(model, filename)
    print(f"Model disimpan ke {filename}")
