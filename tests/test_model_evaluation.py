import pandas as pd # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
from test_model_training import make_dummy_pipeline

def test_accuracy_dummy():
    X = pd.DataFrame({
        "age": [54, 60],
        "sex": [1, 0],
        "cp": [0, 2],
        "trestbps": [140, 130],
        "chol": [239, 250],
        "fbs": [0, 1],
        "restecg": [1, 0],
        "thalch": [160, 150],
        "exang": [0, 1],
        "oldpeak": [1.2, 2.3],
        "slope": [2, 1],
        "ca": [0, 1],
        "thal": [3, 2],
        "dataset": [1, 2]
    })
    y = [0, 1]
    model = make_dummy_pipeline()
    model.fit(X, y)
    acc = accuracy_score(y, model.predict(X))
    assert acc == 1.0, "Akurasi dummy harus 100% pada data latih"
