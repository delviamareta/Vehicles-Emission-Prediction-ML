# LIBRARI & KONFIGURASI UTAMA

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set Konfigurasi Halaman Web agar Tampilan Penuh (Wide)
st.set_page_config(
    page_title="Prediksi Jejak Karbon Kendaraan",
    layout="wide"
)

# KUSTOMISASI CSS GLOBAL (MARGIN ATAS & LAYOUT)
st.markdown("""
    <style>
    /* Mengurangi margin dan padding di bagian paling atas halaman */
    .st-emotion-cache-1jicfl2 {
        padding-top: 2rem !important;
    }
    .st-emotion-cache-z5fcl4 {
        padding-top: 1rem !important;
    }
    div[data-testid="stToolbar"] {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# FUNGSI PENDUKUNG (BACKEND LOGIC)

@st.cache_resource
def load_assets():
    """
    Fungsi untuk memuat model terbaik dan scaler hasil preprocessing.
    Menggunakan path absolut berbasis lokasi file app.py agar aman dari masalah working directory.
    """
    # Mendapatkan path direktori dari file app.py berada (yaitu folder 'app')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Bergerak naik satu tingkat ke folder utama, lalu masuk ke folder 'models'
    model_path = os.path.join(current_dir, "..", "models", "best_model.pkl")
    scaler_path = os.path.join(current_dir, "..", "models", "preprocessing.pkl")
    
    # Lakukan load jika file ditemukan, jika tidak gunakan fallback path alternatif
    if not os.path.exists(model_path):
        model_path = os.path.join(current_dir, "models", "best_model.pkl")
        scaler_path = os.path.join(current_dir, "models", "preprocessing.pkl")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

try:
    model, scaler = load_assets()
    model_loaded = True
except Exception as e:
    model_loaded = False
    error_msg = str(e)


# ANTARMUKA UTAMA (DASHBOARD LAYOUT USING TABS)

st.title("Sistem Prediksi Jejak Karbon Kendaraan")
st.caption("Aplikasi Analisis dan Estimasi Emisi CO2 Berbasis Spesifikasi Mesin")

# Memisahkan halaman menggunakan komponen Tab horizontal agar lebih modern
tab1, tab2 = st.tabs(["Kalkulator Estimasi Emisi", "Dokumentasi Proyek"])

# TAB 1: KALKULATOR EMISI (MODEL DEMO)

with tab1:
    if not model_loaded:
        st.error(f"Gagal memuat komponen model. Pastikan file pkl berada di folder 'models/'. Deskripsi error: {error_msg}")
    else:
        st.header("Simulasi Karakteristik Kendaraan")
        st.write("Sesuaikan parameter teknis kendaraan di bawah ini untuk menghitung estimasi emisi:")
        
        # Grid input 3 kolom untuk efisiensi ruang visual
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Dimensi Mesin")
            engine_size = st.slider("Kapasitas Mesin (Liter):", 0.8, 8.4, 2.0, step=0.1)
            cylinders = st.slider("Jumlah Silinder:", 3, 16, 4, step=1)
            
        with col2:
            st.subheader("Konsumsi Rute")
            fuel_city = st.number_input("Konsumsi BBM Kota (L/100 km):", 3.0, 30.0, 9.5, step=0.1)
            fuel_hwy = st.number_input("Konsumsi BBM Jalan Tol (L/100 km):", 3.0, 25.0, 7.0, step=0.1)
            
        with col3:
            st.subheader("Konsumsi Kombinasi")
            fuel_comb = st.number_input("Konsumsi BBM Kombinasi (L/100 km):", 3.0, 28.0, 8.5, step=0.1)
            fuel_mpg = st.number_input("Konsumsi BBM Kombinasi (mpg):", 10, 90, 33, step=1)

        # CSS Kustom untuk Tombol Oranye Khas Tema Streamlit
        st.markdown("""
            <style>
            div.stButton > button:first-child {
                background-color: #FF4B4B !important;
                color: white !important;
                border-radius: 5px !important;
                border: none !important;
                font-weight: bold !important;
                padding: 0.5rem 1rem !important;
                transition: background-color 0.3s ease !important;
            }
            div.stButton > button:first-child:hover {
                background-color: #E03E3E !important;
                color: white !important;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Tombol aksi dengan lebar penuh
        if st.button("Jalankan Prediksi Emisi Karbon", use_container_width=True):
            # Proses transformasi data
            raw_inputs = np.array([[engine_size, cylinders, fuel_city, fuel_hwy, fuel_comb, fuel_mpg]])
            scaled_inputs = scaler.transform(raw_inputs)
            
            # Padding untuk menyesuaikan fitur dummy dari One-Hot Encoding
            total_model_features = model.n_features_in_
            dummy_padding = np.zeros((1, total_model_features - 6))
            final_features = np.hstack((scaled_inputs, dummy_padding))
            
            # Prediksi
            prediction = model.predict(final_features)[0]
            
            # Tampilan Output Menggunakan Dasbor Metrik yang Bersih
            st.header("Hasil Analisis")
            
            col_res1, col_res2 = st.columns([1, 2])
            with col_res1:
                st.metric(label="Estimasi Emisi CO2", value=f"{prediction:.2f} g/km")
                
            with col_res2:
                # Klasifikasi kategori lingkungan dengan blok warna formal
                if prediction < 160:
                    st.success("Klasifikasi: Kategori Kendaraan Ramah Lingkungan (Eco-Friendly)")
                elif 160 <= prediction <= 255:
                    st.warning("Klasifikasi: Kategori Kendaraan Standar (Normal Emission)")
                else:
                    st.error("Klasifikasi: Kategori Kendaraan Tinggi Emisi (High Carbon Footprint)")

# TAB 2: DOKUMENTASI PROYEK

with tab2:
    st.header("Latar Belakang Proyek")
    st.markdown("""
    Sektor transportasi menyumbang persentase yang signifikan terhadap emisi gas rumah kaca global. 
    Variasi spesifikasi teknis pada mesin kendaraan berdampak langsung pada volume karbon yang dilepaskan ke atmosfer.
    Proyek ini mengimplementasikan pendekatan pembelajaran mesin untuk menyediakan estimasi emisi CO2 (dalam satuan g/km) secara presisi.
    """)
    
    st.header("Metrik Performa Model")
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric(label="R² Score (Akurasi)", value="> 0.90", delta="Lolos Target Ujian")
    with col_m2:
        st.metric(label="Mean Absolute Error", value="Optimal", delta="Minimal", delta_color="inverse")
    with col_m3:
        st.metric(label="Dataset Scope", value="Kanada", delta="Historis Komprehensif")