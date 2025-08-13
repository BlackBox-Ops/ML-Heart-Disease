# build.py (kompatibel dengan train.py revisi)
import numpy as np
import os
import pandas as pd # type: ignore
import joblib # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from sklearn.metrics import ( # type: ignore
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# === 1. Load Model Pipeline Terbaik (.joblib) ===
best_model_path = os.path.join("../models", "svc_best_pipeline.joblib")  # ganti sesuai model pipeline terbaik
best_overall_model_name = "SVC (Best Model - Pipeline)"
model = joblib.load(best_model_path)

# === 2. Load Data Test (mentah) ===
# Data test ini belum dipreprocessing, biarkan pipeline yang memproses
X_test = pd.read_csv("../data/test-data/X_test_raw.csv")
y_test = pd.read_csv("../data/test-data/y_test.csv").squeeze()

# === 3. Prediksi ===
y_pred_final = model.predict(X_test)

# Untuk ROC AUC, cek apakah model punya predict_proba atau decision_function
if hasattr(model, "predict_proba"):
    y_prob_final = model.predict_proba(X_test)[:, 1]
elif hasattr(model, "decision_function"):
    from sklearn.utils.extmath import softmax # type: ignore
    decision_scores = model.decision_function(X_test)
    y_prob_final = softmax(np.c_[1-decision_scores, decision_scores])[:, 1]
else:
    y_prob_final = [0] * len(y_test)

# === 4. Menghitung Metrik Evaluasi ===
accuracy = accuracy_score(y_test, y_pred_final)
precision = precision_score(y_test, y_pred_final)
recall = recall_score(y_test, y_pred_final)
f1 = f1_score(y_test, y_pred_final)
roc_auc = roc_auc_score(y_test, y_prob_final)
conf_matrix = confusion_matrix(y_test, y_pred_final)

print(f"\nMetrik Evaluasi Model Terbaik ({best_overall_model_name}):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)

# === 5. Visualisasi Confusion Matrix ===
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title(f'Confusion Matrix for {best_overall_model_name}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print("\nProses klasifikasi penyakit jantung selesai.")
