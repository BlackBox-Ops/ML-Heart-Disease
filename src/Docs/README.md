# Heart Disease Prediction - Project Documentation

Direktori `src` berisi kode sumber utama untuk membangun, melatih, mendokumentasikan, dan menguji model machine learning untuk prediksi penyakit jantung. Berikut penjelasan tiap file dan paket:

## 📁  Struktur Direktori `src`
### File dan Fungsinya

| File / Paket               | Fungsi Utama                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| <b>api-doc.py</b>              | Dokumentasi API untuk model ML, memudahkan integrasi dengan aplikasi lain. |
| <b>build.py</b>                | Membangun pipeline model ML dari preprocessing hingga siap digunakan.      |
| <b>model_training.py</b>       | Paket utama untuk training model SVM, XGBoost, Random Forest.               |
| <b>preprocessing.py</b>        | Pembersihan data, transformasi, normalisasi, fitur engineering.             |
| <b>test-model-joblib.py</b>    | Menguji model yang sudah tersimpan dalam format `joblib`.                   |
| <b>training-v2.py</b>          | Training Logistic Regression lebih cepat dibanding `train.py`.             |
| <b>utils.py</b>                | Fungsi bantu untuk preprocessing dan modul lain.                             |

---

## Diagram Alur Pipeline

```text
Raw Data
   │
   ▼
preprocessing.py (cleaning, transformasi, scaling)
   │
   ▼
model_training.py / training-v2.py / train.py
   │
   ▼
build.py (menyusun pipeline)
   │
   ▼
Model ML (SVM, XGBoost, Random Forest, Logistic Regression)
   │
   ├──> test-model-joblib.py (uji prediksi)
   │
   └──> api-doc.py (generate dokumentasi API)
```

### 1. `api-doc.py`
- Fungsi: Membuat dokumentasi API untuk model machine learning heart prediction.
- Deskripsi: Menghasilkan endpoint dokumentasi yang memudahkan integrasi model ke sistem lain atau frontend.
- <b>Cara menggunakan</b> 
```python
# keluar direktori jika masih di folder src/Docs 
# cd .. 
# jika sudah di direktori src kemudian ketikan
uvicorn api-doc:app --reload
```
- `--reload` → otomatis me-reload server saat ada perubahan kode
- Akses API di browser: `http://127.0.0.1:8000/docs` untuk tampilan Swagger UI.

### 2. `build.py`
- Fungsi: Membangun model machine learning.
- Deskripsi: Menyusun pipeline training dari preprocessing hingga model siap digunakan.
- <b>Cara menggunakan</b>
```python 
# jika di luar direktori src cukup ketikan
python src/build.py

# tapi kalo misal berada di direktori src ketikan
python build.py
```

### 3. `model_training.py`
- Fungsi: Paket utama untuk training model.
- Deskripsi: Digunakan oleh `train.py` untuk melatih beberapa algoritma:
  - **SVM (Support Vector Machine)**
  - **XGBoost**
  - **Random Forest Classifier**
- Catatan: Mengelola pipeline training dan evaluasi model.

### 4. `preprocessing.py`
- Fungsi: Proses awal data sebelum training.
- Deskripsi: Melakukan pembersihan data (`cleaning`), transformasi, normalisasi, dan fitur engineering untuk meningkatkan performa model.
- <b>Cara menggunakan</b>
```python 
# jika di luar direktori src cukup ketikan
python src/preprocessing.py

# tapi kalo misal berada di direktori src ketikan
python preprocessing.py
```

### 5. `test-model-joblib.py`
- Fungsi: Menguji model yang sudah tersimpan dalam format `joblib`.
- Deskripsi: Memastikan model yang telah disimpan tetap memberikan prediksi yang akurat sebelum digunakan dalam production atau API.
- <b>Cara menggunakan</b>
```python 
# jika di luar direktori src cukup ketikan
python src/test-model-joblib.py

# tapi kalo misal berada di direktori src ketikan
python test-model-joblib.py
```

### 6. `training-v2.py`
- Fungsi: Alternatif training model machine learning dengan Logistic Regression.
- Deskripsi: Melatih model lebih cepat dibandingkan `train.py`, cocok untuk eksperimen atau deployment cepat.
- <b>Cara menggunakan</b>
```python 
# jika di luar direktori src cukup ketikan
python src/training-v2.py

# tapi kalo misal berada di direktori src ketikan
python training-v2.py
```

### 7. `utils.py`
- Fungsi: Fungsi bantu untuk preprocessing dan modul lainnya.
- Deskripsi: Digunakan oleh `preprocessing.py` dan file lain untuk modularisasi kode, seperti scaling, encoding, dan utilities tambahan.

---

## Catatan
- Semua file dikembangkan dengan fokus pada **modularitas**, sehingga setiap bagian pipeline dapat digunakan secara independen.
- Model yang digunakan mencakup **classification algorithms** yang umum pada prediksi penyakit jantung.
- Dokumentasi API (`api-doc.py`) mempermudah integrasi dengan aplikasi web atau mobile.

---

## Contoh Alur Kerja
1. Lakukan preprocessing data dengan `preprocessing.py`.
2. Latih model menggunakan `train.py` atau `training-v2.py`.
3. Simpan model ke format `joblib`.
4. Uji model dengan `test-model-joblib.py`.
5. Buat dokumentasi API menggunakan `api-doc.py` untuk integrasi ke sistem lain.
