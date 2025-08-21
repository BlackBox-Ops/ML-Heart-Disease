# app_streamlit.py
import streamlit as st
import pandas as pd
import joblib
import os
from pydantic import BaseModel, conint, confloat, ValidationError

# ============================================================
# KONFIGURASI
# ============================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models/python-models", "logisticregression_best_pipeline.joblib")

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalch", "exang", "oldpeak", "slope", "ca", "thal"
]

# ============================================================
# MODEL INPUT (Sama dengan app.py)
# ============================================================
class HeartInput(BaseModel):
    age: conint(ge=18, le=100)
    sex: conint(ge=0, le=1)
    cp: conint(ge=0, le=3)
    trestbps: conint(ge=80, le=200)
    chol: conint(ge=100, le=600)
    fbs: conint(ge=0, le=1)
    restecg: conint(ge=0, le=2)
    thalch: conint(ge=60, le=220)
    exang: conint(ge=0, le=1)
    oldpeak: confloat(ge=0, le=6)
    slope: conint(ge=0, le=2)
    ca: conint(ge=0, le=4)
    thal: conint(ge=0, le=3)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

# ============================================================
# MAPPING UNTUK INPUT USER-FRIENDLY
# ============================================================
# Sesuai dengan dataset UCI Heart Disease
OPTION_MAPPINGS = {
    "sex": {
        "Perempuan": 0,
        "Laki-laki": 1
    },
    "cp": {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    },
    "fbs": {
        "Tidak (≤ 120 mg/dl)": 0,
        "Ya (> 120 mg/dl)": 1
    },
    "restecg": {
        "Normal": 0,
        "Memiliki kelainan gelombang ST-T": 1,
        "Hipertrofi ventrikel kiri": 2
    },
    "exang": {
        "Tidak": 0,
        "Ya": 1
    },
    "slope": {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    },
    "ca": {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3
    },
    "thal": {
        "Normal": 0,
        "Fixed Defect": 1,
        "Reversible Defect": 2,
        "Tidak terdeteksi": 3
    }
}

# ============================================================
# FUNGSI UTAMA STREAMLIT
# ============================================================
def main():
    st.set_page_config(
        page_title="Prediksi Penyakit Jantung",
        page_icon="❤️",
        layout="wide"
    )
    
    # Inisialisasi session state untuk menyimpan nilai input
    if 'input_values' not in st.session_state:
        st.session_state.input_values = {
            "age": 50,
            "sex": "Laki-laki",
            "cp": "Typical Angina",
            "trestbps": 120,
            "chol": 200,
            "fbs": "Tidak (≤ 120 mg/dl)",
            "restecg": "Normal",
            "thalch": 150,
            "exang": "Tidak",
            "oldpeak": 1.0,
            "slope": "Upsloping",
            "ca": "0",
            "thal": "Normal"
        }
    
    st.title("Prediksi Penyakit Jantung ❤️")
    st.markdown("Aplikasi ini memprediksi kemungkinan penyakit jantung berdasarkan parameter medis.")
    
    # Tab untuk input manual dan contoh prediksi
    tab1, tab2 = st.tabs(["Input Data Pasien", "Contoh Kasus"])
    
    with tab1:
        # Form input data pasien
        st.header("Formulir Data Medis Pasien")
        
        with st.form("patient_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Data Demografi")
                age = st.number_input("Usia (tahun)", min_value=18, max_value=100, value=st.session_state.input_values["age"], step=1, help="Masukkan usia antara 18-100 tahun")
                sex = st.radio("Jenis Kelamin", list(OPTION_MAPPINGS["sex"].keys()), 
                              index=list(OPTION_MAPPINGS["sex"].keys()).index(st.session_state.input_values["sex"]))
            
            with col2:
                st.subheader("Parameter Klinis")
                trestbps = st.number_input("Tekanan Darah (mm Hg)", min_value=80, max_value=200, value=st.session_state.input_values["trestbps"], step=1, help="Tekanan darah sistolik saat istirahat")
                chol = st.number_input("Kolesterol Serum (mg/dl)", min_value=100, max_value=600, value=st.session_state.input_values["chol"], step=1, help="Nilai kolesterol dalam mg/dl")
                thalch = st.number_input("Detak Jantung Maksimum", min_value=60, max_value=220, value=st.session_state.input_values["thalch"], step=1, help="Detak jantung maksimum yang dicapai")
                oldpeak = st.number_input("Depresi ST", min_value=0.0, max_value=6.0, value=st.session_state.input_values["oldpeak"], step=0.1, help="Depresi ST yang diinduksi olahraga relatif terhadap istirahat")
            
            st.subheader("Parameter Spesifik")
            col3, col4 = st.columns(2)
            
            with col3:
                cp = st.selectbox("Jenis Nyeri Dada", list(OPTION_MAPPINGS["cp"].keys()), 
                                 index=list(OPTION_MAPPINGS["cp"].keys()).index(st.session_state.input_values["cp"]),
                                 help="Jenis nyeri dada yang dialami")
                fbs = st.radio("Gula Darah Puasa", list(OPTION_MAPPINGS["fbs"].keys()),
                              index=list(OPTION_MAPPINGS["fbs"].keys()).index(st.session_state.input_values["fbs"]),
                              help="Gula darah puasa > 120 mg/dl")
                restecg = st.selectbox("Hasil Elektrokardiografi", list(OPTION_MAPPINGS["restecg"].keys()),
                                      index=list(OPTION_MAPPINGS["restecg"].keys()).index(st.session_state.input_values["restecg"]),
                                      help="Hasil EKG saat istirahat")
            
            with col4:
                exang = st.radio("Induksi Angina", list(OPTION_MAPPINGS["exang"].keys()),
                                index=list(OPTION_MAPPINGS["exang"].keys()).index(st.session_state.input_values["exang"]),
                                help="Angina yang diinduksi oleh olahraga")
                slope = st.selectbox("Slope Puncak ST", list(OPTION_MAPPINGS["slope"].keys()),
                                   index=list(OPTION_MAPPINGS["slope"].keys()).index(st.session_state.input_values["slope"]),
                                   help="Slope segmen ST puncak latihan")
                ca = st.selectbox("Jumlah Pembuluh Darah Utama", list(OPTION_MAPPINGS["ca"].keys()),
                                 index=list(OPTION_MAPPINGS["ca"].keys()).index(st.session_state.input_values["ca"]),
                                 help="Jumlah pembuluh darah utama yang diwarnai fluorosopi")
                thal = st.selectbox("Thalassemia", list(OPTION_MAPPINGS["thal"].keys()),
                                   index=list(OPTION_MAPPINGS["thal"].keys()).index(st.session_state.input_values["thal"]),
                                   help="Hasil tes thalassemia")
            
            # Tombol submit
            submitted = st.form_submit_button("Lakukan Prediksi", type="primary")
            
            if submitted:
                # Update session state dengan nilai terbaru
                st.session_state.input_values = {
                    "age": age,
                    "sex": sex,
                    "cp": cp,
                    "trestbps": trestbps,
                    "chol": chol,
                    "fbs": fbs,
                    "restecg": restecg,
                    "thalch": thalch,
                    "exang": exang,
                    "oldpeak": oldpeak,
                    "slope": slope,
                    "ca": ca,
                    "thal": thal
                }
                
                # Membuat dictionary input untuk model
                input_dict = {
                    "age": age,
                    "sex": OPTION_MAPPINGS["sex"][sex],
                    "cp": OPTION_MAPPINGS["cp"][cp],
                    "trestbps": trestbps,
                    "chol": chol,
                    "fbs": OPTION_MAPPINGS["fbs"][fbs],
                    "restecg": OPTION_MAPPINGS["restecg"][restecg],
                    "thalch": thalch,
                    "exang": OPTION_MAPPINGS["exang"][exang],
                    "oldpeak": oldpeak,
                    "slope": OPTION_MAPPINGS["slope"][slope],
                    "ca": OPTION_MAPPINGS["ca"][ca],
                    "thal": OPTION_MAPPINGS["thal"][thal]
                }
                
                # Validasi input
                try:
                    heart_input = HeartInput(**input_dict)
                except ValidationError as e:
                    st.error(f"Error validasi input: {e}")
                    return
                
                # Prediksi
                try:
                    df = pd.DataFrame([input_dict], columns=FEATURE_NAMES)
                    prediction = model.predict(df)[0]
                    probability = model.predict_proba(df)[0] if hasattr(model, "predict_proba") else None
                    
                    # Tampilkan hasil
                    st.subheader("Hasil Prediksi")
                    
                    if prediction == 1:
                        st.error("**Hasil: Terdeteksi Penyakit Jantung** ❌")
                        st.warning("⚠️ Hasil ini menunjukkan kemungkinan adanya penyakit jantung. Silakan konsultasi dengan dokter untuk diagnosis lebih lanjut.")
                    else:
                        st.success("**Hasil: Tidak Terdeteksi Penyakit Jantung** ✅")
                    
                    if probability is not None:
                        # Tampilkan probabilitas untuk kedua kelas
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Probabilitas Tidak Sakit", f"{probability[0]*100:.2f}%")
                        with col2:
                            st.metric("Probabilitas Sakit", f"{probability[1]*100:.2f}%")
                        
                        # Visualisasi probabilitas
                        st.progress(probability[1], text="Tingkat Risiko Penyakit Jantung")
                    
                except Exception as e:
                    st.error(f"Error dalam prediksi: {e}")
    
    with tab2:
        st.header("Contoh Kasus untuk Prediksi")
        
        # Contoh kasus dengan risiko tinggi
        st.subheader("Kasus dengan Risiko Tinggi")
        st.markdown("""
        Berikut adalah contoh nilai-nilai yang cenderung menghasilkan prediksi penyakit jantung:
        - Usia: 65 tahun
        - Jenis Kelamin: Laki-laki
        - Jenis Nyeri Dada: Asymptomatic
        - Tekanan Darah: 180 mm Hg
        - Kolesterol: 300 mg/dl
        - Gula Darah Puasa: Ya (> 120 mg/dl)
        - Hasil EKG: Hipertrofi ventrikel kiri
        - Detak Jantung Maksimum: 90
        - Induksi Angina: Ya
        - Depresi ST: 4.0
        - Slope Puncak ST: Downsloping
        - Jumlah Pembuluh Darah: 3
        - Thalassemia: Fixed Defect
        """)
        
        if st.button("Coba Kasus Risiko Tinggi", key="high_risk"):
            # Set nilai-nilai risiko tinggi di session state
            st.session_state.input_values = {
                "age": 65,
                "sex": "Laki-laki",
                "cp": "Asymptomatic",
                "trestbps": 180,
                "chol": 300,
                "fbs": "Ya (> 120 mg/dl)",
                "restecg": "Hipertrofi ventrikel kiri",
                "thalch": 90,
                "exang": "Ya",
                "oldpeak": 4.0,
                "slope": "Downsloping",
                "ca": "3",
                "thal": "Fixed Defect"
            }
            
            st.success("Nilai risiko tinggi telah dimasukkan. Silakan beralih ke tab 'Input Data Pasien' dan klik 'Lakukan Prediksi'.")
            st.rerun()
        
        # Contoh kasus dengan risiko rendah
        st.subheader("Kasus dengan Risiko Rendah")
        st.markdown("""
        Berikut adalah contoh nilai-nilai yang cenderung menghasilkan prediksi tidak ada penyakit jantung:
        - Usia: 45 tahun
        - Jenis Kelamin: Perempuan
        - Jenis Nyeri Dada: Typical Angina
        - Tekanan Darah: 120 mm Hg
        - Kolesterol: 180 mg/dl
        - Gula Darah Puasa: Tidak (≤ 120 mg/dl)
        - Hasil EKG: Normal
        - Detak Jantung Maksimum: 150
        - Induksi Angina: Tidak
        - Depresi ST: 0.5
        - Slope Puncak ST: Upsloping
        - Jumlah Pembuluh Darah: 0
        - Thalassemia: Normal
        """)
        
        if st.button("Coba Kasus Risiko Rendah", key="low_risk"):
            # Set nilai-nilai risiko rendah di session state
            st.session_state.input_values = {
                "age": 45,
                "sex": "Perempuan",
                "cp": "Typical Angina",
                "trestbps": 120,
                "chol": 180,
                "fbs": "Tidak (≤ 120 mg/dl)",
                "restecg": "Normal",
                "thalch": 150,
                "exang": "Tidak",
                "oldpeak": 0.5,
                "slope": "Upsloping",
                "ca": "0",
                "thal": "Normal"
            }
            
            st.success("Nilai risiko rendah telah dimasukkan. Silakan beralih ke tab 'Input Data Pasien' dan klik 'Lakukan Prediksi'.")
            st.rerun()
    
    # Informasi tentang dataset
    with st.expander("Informasi tentang Dataset"):
        st.markdown("""
        Dataset yang digunakan adalah **UCI Heart Disease Dataset** yang berisi parameter medis pasien.
        
        **Keterangan Fitur:**
        - **age**: Usia dalam tahun
        - **sex**: Jenis kelamin (0 = perempuan, 1 = laki-laki)
        - **cp**: Jenis nyeri dada
          - 0 = Typical Angina
          - 1 = Atypical Angina  
          - 2 = Non-anginal Pain
          - 3 = Asymptomatic
        - **trestbps**: Tekanan darah saat istirahat (mm Hg)
        - **chol**: Kolesterol serum (mg/dl)
        - **fbs**: Gula darah puasa > 120 mg/dl (0 = tidak, 1 = ya)
        - **restecg**: Hasil elektrokardiografi saat istirahat
          - 0 = Normal
          - 1 = Memiliki kelainan gelombang ST-T
          - 2 = Hipertrofi ventrikel kiri
        - **thalch**: Detak jantung maksimum
        - **exang**: Angina yang diinduksi oleh olahraga (0 = tidak, 1 = ya)
        - **oldpeak**: Depresi ST yang diinduksi oleh olahraga relatif terhadap istirahat
        - **slope**: Slope segmen ST puncak latihan
          - 0 = Upsloping
          - 1 = Flat
          - 2 = Downsloping
        - **ca**: Jumlah pembuluh darah utama (0-3) yang diwarnai oleh fluorosopi
        - **thal**: Kelainan thalassemia
          - 0 = Normal
          - 1 = Fixed Defect
          - 2 = Reversible Defect
          - 3 = Tidak terdeteksi
        """)

if __name__ == "__main__":
    main()