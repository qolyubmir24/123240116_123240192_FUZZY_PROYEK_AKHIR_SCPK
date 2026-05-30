import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ==========================================
# 1. KONFIGURASI AWAL
# ==========================================
st.set_page_config(page_title="SPK Fuzzy Mamdani", page_icon="🌾", layout="wide")

DATASET_FILE = "Data_Tanaman_Padi_Sumatera.csv"

KRITERIA_COLS = ["Produksi", "Luas Panen", "Curah hujan", "Kelembapan", "Suhu rata-rata"]
REQUIRED_COLS = ["Provinsi"] + KRITERIA_COLS

def clean_numeric_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    def normalize(v: str) -> str:
        if v.lower() in {"", "nan", "none", "null"}: return ""
        v = v.replace(" ", "")
        return v.replace(".", "").replace(",", ".") if "," in v and "." in v else v.replace(",", ".")
    return pd.to_numeric(s.map(normalize), errors="coerce")

def label_kelayakan(skor: float) -> str:
    if skor >= 70: return "✅ Layak"
    elif skor >= 40: return "⚠️ Cukup Layak"
    return "❌ Kurang Layak"

@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Langsung membaca file CSV tanpa os.path
    df_raw = pd.read_csv(DATASET_FILE)
    
    for col in KRITERIA_COLS:
        df_raw[col] = clean_numeric_series(df_raw[col])
    
    df_raw = df_raw.dropna(subset=REQUIRED_COLS).copy()
    
    # Agregasi data per Provinsi
    df_agg = (df_raw.groupby("Provinsi", as_index=False)[KRITERIA_COLS]
              .mean(numeric_only=True).sort_values("Provinsi").reset_index(drop=True))
    
    return df_raw, df_agg

try:
    df_raw, df = load_data()
except Exception as e:
    st.error(str(e))
    st.stop()

# ==========================================
# 2. SETUP SISTEM FUZZY (SCIKIT-FUZZY)
# ==========================================
produksi = ctrl.Antecedent(np.linspace(df["Produksi"].min(), df["Produksi"].max(), 100), 'Produksi')
luas_panen = ctrl.Antecedent(np.linspace(df["Luas Panen"].min(), df["Luas Panen"].max(), 100), 'Luas Panen')
curah_hujan = ctrl.Antecedent(np.linspace(df["Curah hujan"].min(), df["Curah hujan"].max(), 100), 'Curah hujan')
kelembapan = ctrl.Antecedent(np.linspace(df["Kelembapan"].min(), df["Kelembapan"].max(), 100), 'Kelembapan')
suhu = ctrl.Antecedent(np.linspace(df["Suhu rata-rata"].min(), df["Suhu rata-rata"].max(), 100), 'Suhu rata-rata')

kelayakan = ctrl.Consequent(np.arange(0, 101, 1), 'Kelayakan')

list_variabel = [produksi, luas_panen, curah_hujan, kelembapan, suhu]

for col, var in zip(KRITERIA_COLS, list_variabel):
    mn, mx, me = df[col].min(), df[col].max(), df[col].mean()
    var['rendah'] = fuzz.trimf(var.universe, [mn, mn, me])
    var['sedang'] = fuzz.trimf(var.universe, [mn, me, mx])
    var['tinggi'] = fuzz.trimf(var.universe, [me, mx, mx])

kelayakan['rendah'] = fuzz.trimf(kelayakan.universe, [0, 0, 50])
kelayakan['sedang'] = fuzz.trimf(kelayakan.universe, [0, 50, 100])
kelayakan['tinggi'] = fuzz.trimf(kelayakan.universe, [50, 100, 100])

rules = [
    ctrl.Rule(produksi['tinggi'] & luas_panen['tinggi'], kelayakan['tinggi']),
    ctrl.Rule(produksi['tinggi'] & curah_hujan['sedang'] & kelembapan['sedang'], kelayakan['tinggi']),
    ctrl.Rule(produksi['tinggi'] & suhu['sedang'], kelayakan['tinggi']),
    ctrl.Rule(produksi['sedang'] & luas_panen['sedang'], kelayakan['sedang']),
    ctrl.Rule(curah_hujan['tinggi'] | kelembapan['tinggi'], kelayakan['sedang']),
    ctrl.Rule(produksi['tinggi'] & luas_panen['sedang'], kelayakan['sedang']),
    ctrl.Rule(produksi['rendah'], kelayakan['rendah']),
    ctrl.Rule(luas_panen['rendah'], kelayakan['rendah']),
    ctrl.Rule(produksi['rendah'] & luas_panen['rendah'], kelayakan['rendah']),
    ctrl.Rule(suhu['tinggi'] & curah_hujan['rendah'], kelayakan['rendah'])
]

kelayakan_ctrl = ctrl.ControlSystem(rules)
mesin_fuzzy = ctrl.ControlSystemSimulation(kelayakan_ctrl)

# ==========================================
# 3. TAMPILAN ANTARMUKA STREAMLIT
# ==========================================
halaman = st.sidebar.radio("Navigasi", ["🏠 Profil Kelompok", "📊 Dataset", "🧠 Hitung SPK Fuzzy"])

if halaman == "🏠 Profil Kelompok":
    st.title("👨‍💻 Profil Anggota Kelompok")
    st.caption("Proyek Akhir Praktikum Sistem Pendukung Keputusan 2025/2026")
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("### Anggota 1\n**Nama:** Mirwan Qolyubi\n\n**NIM:** 123240116\n\n**Peran:** Backend & Fuzzy Logic")
    with col2:
        st.success("### Anggota 2\n**Nama:** Aulia Cahaya R.\n\n**NIM:** 123240192\n\n**Peran:** Frontend & Data Analyst")

elif halaman == "📊 Dataset":
    st.title("📊 Dataset Tanaman Padi — Sumatera")
    
    jumlah_baris = len(df_raw)
    if jumlah_baris >= 250:
        st.success(f"✅ Syarat Dataset Terpenuhi: Memiliki {jumlah_baris} baris data (Minimal 250).")
    else:
        st.error(f"❌ Syarat Dataset Belum Terpenuhi: Hanya memiliki {jumlah_baris} baris data dari batas minimal 250.")
        
    st.markdown("### Tabel Data Mentah")
    st.dataframe(df_raw, use_container_width=True)

    st.markdown("### Tabel Agregasi (Rata-rata per Alternatif/Provinsi)")
    st.dataframe(df.round(2), use_container_width=True)

elif halaman == "🧠 Hitung SPK Fuzzy":
    st.title("🧠 Simulasi & Hitung SPK Fuzzy Mamdani")
    
    with st.expander("📈 Lihat Proses SPK: Kurva Keanggotaan (Membership Functions)", expanded=False):
        pilihan_kurva = st.selectbox("Pilih Kriteria untuk divisualisasikan:", KRITERIA_COLS + ["Kelayakan (Output)"])
        fig, ax = plt.subplots(figsize=(8, 3))
        
        if pilihan_kurva == "Kelayakan (Output)":
            ax.plot(kelayakan.universe, kelayakan['rendah'].mf, label='Rendah', lw=2)
            ax.plot(kelayakan.universe, kelayakan['sedang'].mf, label='Sedang', lw=2)
            ax.plot(kelayakan.universe, kelayakan['tinggi'].mf, label='Tinggi', lw=2)
        else:
            idx = KRITERIA_COLS.index(pilihan_kurva)
            var = list_variabel[idx]
            ax.plot(var.universe, var['rendah'].mf, label='Rendah', lw=2)
            ax.plot(var.universe, var['sedang'].mf, label='Sedang', lw=2)
            ax.plot(var.universe, var['tinggi'].mf, label='Tinggi', lw=2)
            
        ax.set_title(f"Kurva Keanggotaan: {pilihan_kurva}")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.markdown("### 🎛️ Simulasi Nilai Kriteria")
    st.caption("Pilih provinsi dan geser slider untuk mensimulasikan perubahan nilai secara instan.")
    
    provinsi_terpilih = st.selectbox("📌 1. Pilih Alternatif Provinsi:", df["Provinsi"].tolist())
    baris_terpilih = df[df["Provinsi"] == provinsi_terpilih].iloc[0]
    
    st.markdown("**⚙️ 2. Atur Simulasi (Geser Slider):**")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        simulasi_produksi = st.slider(
            "🌾 Produksi", 
            min_value=float(df["Produksi"].min()), 
            max_value=float(df["Produksi"].max()), 
            value=float(baris_terpilih["Produksi"])
        )
    with c2:
        simulasi_luas = st.slider(
            "📐 Luas Panen", 
            min_value=float(df["Luas Panen"].min()), 
            max_value=float(df["Luas Panen"].max()), 
            value=float(baris_terpilih["Luas Panen"])
        )
    with c3:
        simulasi_curah = st.slider(
            "🌧️ Curah Hujan", 
            min_value=float(df["Curah hujan"].min()), 
            max_value=float(df["Curah hujan"].max()), 
            value=float(baris_terpilih["Curah hujan"])
        )

    if st.button("🚀 Eksekusi Perhitungan SPK Keseluruhan", type="primary", use_container_width=True):
        hasil = []
        
        for _, baris in df.iterrows():
            if baris["Provinsi"] == provinsi_terpilih:
                mesin_fuzzy.input['Produksi'] = simulasi_produksi
                mesin_fuzzy.input['Luas Panen'] = simulasi_luas
                mesin_fuzzy.input['Curah hujan'] = simulasi_curah
            else:
                mesin_fuzzy.input['Produksi'] = baris['Produksi']
                mesin_fuzzy.input['Luas Panen'] = baris['Luas Panen']
                mesin_fuzzy.input['Curah hujan'] = baris['Curah hujan']
                
            mesin_fuzzy.input['Kelembapan'] = baris['Kelembapan']
            mesin_fuzzy.input['Suhu rata-rata'] = baris['Suhu rata-rata']
            
            mesin_fuzzy.compute()
            skor_akhir = mesin_fuzzy.output['Kelayakan']
            
            hasil.append({
                "Provinsi": baris["Provinsi"] + (" (Simulasi)" if baris["Provinsi"] == provinsi_terpilih else ""),
                "Skor Akhir": round(skor_akhir, 2),
                "Status": label_kelayakan(skor_akhir)
            })

        df_hasil = pd.DataFrame(hasil).sort_values("Skor Akhir", ascending=False).reset_index(drop=True)
        df_hasil.index += 1
        df_hasil.index.name = "Peringkat"
        
        st.success("✅ Eksekusi selesai! Hasil telah diperbarui.")
        
        st.subheader("🏆 Tabel Hasil Perangkingan Akhir")
        st.dataframe(df_hasil, use_container_width=True)

        st.subheader("📊 Visualisasi Hasil Akhir")
        fig2, ax2 = plt.subplots(figsize=(9, max(3, len(df_hasil) * 0.45)))
        bars = ax2.barh(df_hasil["Provinsi"][::-1], df_hasil["Skor Akhir"][::-1], color='teal', alpha=0.8)
        ax2.set_xlabel("Skor Kelayakan Defuzzifikasi (0–100)")
        ax2.set_xlim(0, 115)
        for bar in bars:
            ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.2f}", va="center", fontsize=9)
        st.pyplot(fig2)
        plt.close(fig2)