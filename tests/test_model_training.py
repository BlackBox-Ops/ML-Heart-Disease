import pandas as pd                                             # type: ignore
from sklearn.pipeline import Pipeline                           # type: ignore
from sklearn.svm import SVC                                     # type: ignore
from sklearn.compose import ColumnTransformer                   # type: ignore
from sklearn.preprocessing import StandardScaler, OneHotEncoder # type: ignore
from sklearn.impute import SimpleImputer                        # type: ignore

def make_dummy_pipeline():
    num_cols = ["age","trestbps","chol","thalch","oldpeak","ca"]
    cat_cols = ["sex","cp","fbs","restecg","exang","slope","thal","dataset"]

    num_trans = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
    cat_trans = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer([('num', num_trans, num_cols), ('cat', cat_trans, cat_cols)])
    return Pipeline([('preprocessor', preprocessor), ('classifier', SVC(probability=True))])

def test_pipeline_training():
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
    preds = model.predict(X)
    assert len(preds) == 2, "Prediksi jumlahnya tidak sesuai"