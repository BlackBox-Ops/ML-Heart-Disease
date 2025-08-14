# Dokumentasi Data

Dokumentasi ini menjelaskan struktur dan isi dari direktori data yang digunakan dalam proyek ini. Data dibagi menjadi beberapa folder untuk memudahkan pengelolaan dan pemrosesan.


### Penjelasan Folder

1. **Folder `raw`**:
   - Berisi data mentah yang belum diproses.
   - Data dalam format CSV (Comma-Separated Values) yang dapat digunakan untuk analisis lebih lanjut.

2. **Folder `processed`**:
   - Berisi data yang telah diproses dan dibersihkan.
   - Data ini siap digunakan untuk analisis atau pelatihan model.

3. **Folder `train`**:
   - Berisi data untuk pelatihan model.
   - `x_train.csv`: Fitur-fitur yang digunakan untuk melatih model.
   - `y_train.csv`: Label atau target yang sesuai dengan fitur dalam `x_train.csv`.

4. **Folder `test`**:
   - Berisi data untuk pengujian model.
   - `x_test.csv`: Fitur-fitur yang digunakan untuk menguji model.
   - `y_test.csv`: Label atau target yang sesuai dengan fitur dalam `x_test.csv`.

## Cara Menggunakan Data

1. **Mengakses Data**:
   - Anda dapat mengakses data mentah dari folder `raw` untuk analisis awal.
   - Gunakan data dalam folder `processed` untuk pelatihan model.

2. **Pelatihan Model**:
   - Gunakan file `x_train.csv` dan `y_train.csv` untuk melatih model Anda.
   - Setelah model dilatih, gunakan `x_test.csv` dan `y_test.csv` untuk menguji kinerja model.

.


