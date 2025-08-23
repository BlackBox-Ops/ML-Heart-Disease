# Heart Disease Prediction Project

Proyek ini bertujuan membangun sistem prediksi penyakit jantung berbasis **Machine Learning** dengan integrasi ke aplikasi **Flask** dan **Streamlit**.  
Struktur proyek dibuat modular agar mudah digunakan, dikembangkan, dan di-deploy.

---

## 📂 Struktur Direktori

| Path               | Deskripsi                                                                 |
|--------------------|---------------------------------------------------------------------------|
| `app/`             | Aplikasi web berbasis **Flask** untuk menjalankan model ML terapan.       |
| `data/`            | Dataset untuk training & testing.                                         |
| ├── `raw/`         | Data mentah.                                                              |
| ├── `cleaned/`     | Data yang sudah dibersihkan.                                              |
| ├── `train/`       | Data latih siap pakai.                                                    |
| └── `sample.json`  | Contoh input JSON untuk uji coba API.                                     |
| `img/`             | Screenshot dokumentasi API & contoh error code.                           |
| `models/`          | Model machine learning yang sudah dilatih (`.joblib` / `.pkl`).           |
| `note.txt`         | Catatan tambahan terkait eksperimen atau dokumentasi internal.            |
| `src/`             | Kode Python utama (pipeline build & training model ML).                   |
| `test/`            | Script testing model.                                                     |
| `main.py`          | Aplikasi **Streamlit** untuk visualisasi & interaksi.                     |
| `run_model.sh`     | Shell script untuk menjalankan pipeline secara otomatis.                  |
| `requirements.txt` | Daftar dependency Python yang diperlukan.                                 |
| `docker-compose.yml` | Konfigurasi Docker Compose untuk menjalankan aplikasi dengan mudah.      |

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
```

### 2. Install Dependencies (Opsional, jika tidak pakai Docker)
```bash
pip install -r requirements.txt
```

### 3. Jalankan Streamlit App (Opsional)
```bash
streamlit run main.py
```

### 4. Jalankan Flask API Secara Manual (Opsional)
Masuk ke folder app/ lalu jalankan:
```bash
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

---

## 🐳 Jalankan dengan Docker Compose (Rekomendasi)

Cara termudah untuk menjalankan aplikasi ini adalah menggunakan **Docker Compose**.  
Pastikan Docker & Docker Compose sudah terinstall di sistem Anda.

```bash
# mengatifkan docker
docker-compose up 

# mematikan docker 
docker-compose down 
```

Aplikasi akan berjalan di [http://localhost:5000](http://localhost:5000)

[![Docker Compose](https://img.shields.io/badge/docker--compose-ready-blue)](https://docs.docker.com/compose/)

---

### Note 
Untuk fitur profile pasien untuk saat ini masih coming soon  
dan akan dilanjutkan di masa mendatang.

---

**Kontribusi sangat terbuka!**  
Silakan fork, buat pull request, atau diskusi di issues