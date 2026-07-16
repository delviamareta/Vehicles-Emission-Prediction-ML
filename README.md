# Vehicle CO2 Emission Prediction Project

- **Nama:** Delvia Mareta Dealova
- **NIM:** A11.2024.15658
- **Mata Kuliah:** Pembelajaran Mesin - Universitas Dian Nuswantoro

## Ringkasan Proyek
Proyek ini bertujuan untuk memprediksi emisi karbon dioksida (CO2) kendaraan berbasis spesifikasi mesin menggunakan pendekatan machine learning regresi. Model terbaik dioptimasi menggunakan algoritma XGBoost/Random Forest hingga mencapai nilai akurasi R² Score >= 0.90.

## Struktur Repositori
```
Vehicles Emission/
├── data/
│   ├── raw/                  <-- Dataset mentah (CO2 Emissions_Canada.csv)
│   └── processed/            <-- Dataset bersih hasil prapemrosesan (clean_data.pkl)
├── notebooks/
│   ├── 01_eda.ipynb          <-- Analisis eksploratif data dan visualisasi insight
│   ├── 02_modeling.ipynb     <-- Pelatihan, tuning, dan evaluasi model regresi
│   └── 03_interpretation.ipynb <-- Interpretasi model menggunakan framework SHAP
├── models/
│   ├── best_model.pkl        <-- File model final yang telah dilatih
│   └── preprocessing.pkl     <-- File scaler untuk transformasi data input
├── app/
│   └── app.py                <-- Script antarmuka aplikasi web Streamlit
├── reports/
│   ├── insight1_dist.png
│   ├── insight2_scatter.png
│   ├── insight3_heatmap.png
│   ├── insight4_fuel.png
│   ├── insight5_boxplot.png
│   └── shap_summary.png
├── requirements.txt          <-- Daftar pustaka pendukung ekosistem proyek
└── README.md                 <-- Dokumentasi utama proyek
```

## Cara Menjalankan Aplikasi Secara Lokal
1. Pastikan library pendukung sudah terpasang:
   ```bash
   pip install -r requirements.txt
   ```
2. Jalankan aplikasi Streamlit dari direktori utama:
   ```bash
   streamlit run app/app.py
   ```
3. Buka browser dan akses URL yang ditampilkan di terminal (biasanya `http://localhost:8501`).

## Fitur Utama Aplikasi
- **Kalkulator Estimasi Emisi**: Input spesifikasi kendaraan (kapasitas mesin, jumlah silinder, konsumsi BBM) untuk mendapatkan estimasi emisi CO2.
- **Klasifikasi Kategori**: Hasil prediksi diklasifikasikan menjadi tiga kategori:
  - Eco-Friendly (< 160 g/km)
  - Normal Emission (160 - 255 g/km)
  - High Carbon Footprint (> 255 g/km)
- **Dokumentasi Proyek**: Penjelasan latar belakang dan metrik performa model.

## Teknologi yang Digunakan
- **Bahasa Pemrograman:** Python
- **Framework ML:** Scikit-Learn, XGBoost
- **Framework Web:** Streamlit
- **Analisis Data:** Pandas, NumPy
- **Visualisasi:** Matplotlib, Seaborn
- **Interpretasi Model:** SHAP

## Deploy ke Streamlit Cloud
1. Push repositori ini ke GitHub (pastikan semua file `.pkl` sudah di-commit).
2. Buka [Streamlit Cloud](https://share.streamlit.io) dan login menggunakan akun GitHub.
3. Klik **New app** → pilih branch, isi path file `app/app.py`.
4. Klik **Deploy**. Streamlit akan otomatis membaca `requirements.txt` dan menjalankan aplikasi.
5. Setelah deploy selesai, salin URL yang diberikan untuk mengakses aplikasi secara online.

## Metrik Evaluasi Model
- **R² Score:** > 0.90 (Target lolos ujian)
- **Mean Absolute Error (MAE):** Minimal
- **Dataset:** CO2 Emissions_Canada (Data historis kendaraan di Kanada)
