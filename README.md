# Heart Disease Prediction Project

Proyek ini bertujuan membangun sistem prediksi penyakit jantung berbasis **Machine Learning** dengan integrasi ke aplikasi **Flask** dan **Streamlit**.  
Struktur proyek dibuat modular agar mudah digunakan, dikembangkan, dan di-deploy.

---

## 📂 Struktur Direktori

├── app/ # Aplikasi web berbasis Flask untuk menjalankan model ML terapan
├── data/ # Dataset untuk training & testing
│ ├── raw/ # Data mentah
│ ├── cleaned/ # Data yang sudah dibersihkan
│ ├── train/ # Data latih siap pakai
│ └── sample.json # Contoh input JSON untuk uji coba API
├── img/ # Screenshot dokumentasi API & error code
├── models/ # Model machine learning yang sudah dilatih (format joblib/pkl)
├── note.txt # Catatan tambahan terkait eksperimen atau dokumentasi internal
├── src/ # Kode Python utama (pipeline build & training model ML)
├── test/ # Script testing model
├── main.py # Aplikasi Streamlit untuk visualisasi & interaksi
├── run_model.sh # Shell script untuk menjalankan pipeline secara otomatis
├── requirements.txt # Dependency Python yang diperlukan


---

## 🚀 Fitur Utama

1. **Aplikasi Flask (app/)**  
   - Menyediakan endpoint API untuk prediksi penyakit jantung.  
   - Mendukung input berupa JSON (contoh: `data/sample.json`).  
   - Dokumentasi API dapat diakses melalui Swagger UI.

2. **Data Management (data/)**  
   - Dataset mentah → dibersihkan → diproses → siap untuk training.  
   - Tersedia **data latih** yang langsung dapat digunakan.  

3. **Model (models/)**  
   - Menyimpan model ML yang sudah dilatih (SVM, XGBoost, Random Forest, Logistic Regression, dll).  
   - Format penyimpanan: `.joblib` atau `.pkl`.  

4. **Pipeline & Training (src/)**  
   - Script untuk preprocessing, training, build pipeline, dan dokumentasi API.  
   - Modular dan reusable.  

5. **Testing (test/)**  
   - Menguji model yang sudah jadi.  
   - Validasi terhadap prediksi dan stabilitas model.  

6. **Visualisasi & UI (main.py)**  
   - Aplikasi berbasis **Streamlit** untuk interaksi dan visualisasi hasil prediksi.  

7. **Automation (run_model.sh)**  
   - Script untuk menjalankan pipeline preprocessing → training → testing → API secara otomatis.  

8. **Dokumentasi (img/ & note/)**  
   - Berisi screenshot dokumentasi API dan kode error.  
   - Folder note untuk catatan eksperimen, arsitektur, atau referensi tambahan.  

---

## ⚡ Cara Menjalankan

### 1. Clone Repository
```bash
git clone <repo-url>
cd heart-disease-prediction

pip install -r requirements.txt
```

### 2. Install Dependencies
```python
pip install -r requirements.txt
```

### 3. Jalankan Streamlit App
```python
streamlit run main.py
```

### 4. Jalankan Flask API
Masuk ke folder app/ lalu jalankan:
```python
cd app
python app.py
```
Atau gunakan Uvicorn untuk API documentation:
```bash
uvicorn src/api-doc:app --reload
```

### 5. Jalankan Pipeline Lengkap
Gunakan script otomatis:
```bash
./run_model.sh
```