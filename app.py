# ============================================================================
# SIMULASI MONTE CARLO — ESTIMASI WAKTU PEMBANGUNAN GEDUNG FITE 5 LANTAI
# [11S1221] Pemodelan dan Simulasi (MODSIM) — Praktikum 5
# ============================================================================
# Cara menjalankan:
#   pip install streamlit plotly numpy pandas scipy
#   streamlit run app.py
# ============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# KONFIGURASI HALAMAN
# ============================================================================
st.set_page_config(
    page_title="MODSIM — Konstruksi FITE",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
}

.stApp {
    background-color: #F4F1EB;
    color: #1a1a1a;
}

.block-container {
    max-width: 1280px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* ─── HEADER INDUSTRIAL ─── */
.top-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    border-top: 5px solid #1a1a1a;
    border-bottom: 2px solid #1a1a1a;
    padding: 1.2rem 0;
    margin-bottom: 1.8rem;
    background: #F4F1EB;
}
.top-header-left .label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    color: #888;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}
.top-header-left h1 {
    font-size: 1.7rem;
    font-weight: 700;
    color: #1a1a1a;
    margin: 0;
    line-height: 1.15;
    letter-spacing: -0.03em;
}
.top-header-right {
    text-align: right;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #888;
    line-height: 2;
}

/* ─── METRIC BOXES ─── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0;
    border: 2px solid #1a1a1a;
    margin-bottom: 1.8rem;
}
.metric-box {
    padding: 1rem 1.2rem;
    border-right: 2px solid #1a1a1a;
    background: #fff;
}
.metric-box:last-child { border-right: none; }
.metric-box .m-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 0.3rem;
}
.metric-box .m-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #1a1a1a;
    line-height: 1;
}
.metric-box .m-unit {
    font-size: 0.7rem;
    color: #888;
    margin-top: 0.2rem;
}
.accent-red  { border-top: 3px solid #D63F2E; }
.accent-blue { border-top: 3px solid #2563EB; }
.accent-grn  { border-top: 3px solid #059669; }
.accent-org  { border-top: 3px solid #D97706; }
.accent-prp  { border-top: 3px solid #7C3AED; }

/* ─── SECTION TITLE ─── */
.sec-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #888;
    padding: 0.4rem 0;
    border-bottom: 1px solid #1a1a1a;
    margin: 1.6rem 0 1rem 0;
}

/* ─── DEADLINE TILES ─── */
.dl-grid {
    display: grid;
    grid-template-columns: repeat(7, 1fr);
    gap: 8px;
    margin-bottom: 1.2rem;
}
.dl-tile {
    border: 2px solid #1a1a1a;
    padding: 0.8rem 0.4rem;
    text-align: center;
    background: #fff;
}
.dl-tile .dl-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #888;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
}
.dl-tile .dl-pct {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: #1a1a1a;
}
.dl-ok  { background: #ECFDF5; border-color: #059669; }
.dl-ok .dl-pct { color: #059669; }
.dl-warn { background: #FFFBEB; border-color: #D97706; }
.dl-warn .dl-pct { color: #D97706; }
.dl-bad  { background: #FEF2F2; border-color: #D63F2E; }
.dl-bad .dl-pct { color: #D63F2E; }

/* ─── INFO PANELS ─── */
.info-panel {
    border: 2px solid #1a1a1a;
    background: #fff;
    padding: 1.2rem;
    margin-bottom: 1rem;
}
.info-panel .ip-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #e5e5e5;
}

/* ─── CHIP PROBABILITAS ─── */
.chip {
    display: inline-block;
    padding: 4px 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    border: 2px solid;
    margin: 2px;
}
.chip-ok   { color: #059669; border-color: #059669; background: #ECFDF5; }
.chip-warn { color: #D97706; border-color: #D97706; background: #FFFBEB; }
.chip-bad  { color: #D63F2E; border-color: #D63F2E; background: #FEF2F2; }

/* ─── REKOMENDASI BOX ─── */
.rec-panel {
    background: #1a1a1a;
    color: #F4F1EB;
    padding: 1.2rem 1.4rem;
    margin-top: 0.8rem;
    font-size: 0.9rem;
    line-height: 1.8;
}
.rec-panel b { color: #FCD34D; }

/* ─── CRITICAL CARD ─── */
.crit-card {
    border: 2px solid #1a1a1a;
    background: #fff;
    padding: 1rem;
}
.crit-card .rank {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.15em;
    color: #888;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.crit-card .name {
    font-size: 0.8rem;
    font-weight: 600;
    color: #1a1a1a;
    margin-bottom: 0.4rem;
}

/* ─── SIDEBAR — FIXED ─── */
[data-testid="stSidebar"] {
    background-color: #1a1a1a !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] .stMarkdown div {
    color: #F4F1EB !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #888 !important;
    border-bottom: 1px solid #333 !important;
    padding-bottom: 0.4rem !important;
    margin-bottom: 0.6rem !important;
}
/* Fix slider label overlap */
[data-testid="stSidebar"] [data-testid="stSlider"] label,
[data-testid="stSidebar"] [data-testid="stNumberInput"] label {
    color: #a0a0a0 !important;
    font-size: 0.78rem !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    position: static !important;
}
/* Hide ghost position labels from Streamlit slider */
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMax"] {
    display: none !important;
}
[data-testid="stSidebar"] .stSlider,
[data-testid="stSidebar"] .stNumberInput {
    margin-bottom: 0.6rem !important;
}
/* Fix all text inside sidebar */
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
    color: #c0c0c0 !important;
}
[data-testid="stSidebar"] input {
    background: #2a2a2a !important;
    color: #F4F1EB !important;
    border-color: #444 !important;
}
/* Expander in sidebar */
[data-testid="stSidebar"] details {
    background: #222 !important;
    border: 1px solid #333 !important;
    border-radius: 2px !important;
    margin-bottom: 4px !important;
}
[data-testid="stSidebar"] details summary {
    color: #c0c0c0 !important;
    font-size: 0.75rem !important;
    padding: 0.4rem 0.6rem !important;
}

/* ─── TABS ─── */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 2px solid #1a1a1a !important;
    gap: 0 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #888 !important;
    border: 2px solid transparent !important;
    border-bottom: none !important;
    padding: 0.5rem 1rem !important;
    background: transparent !important;
    border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #1a1a1a !important;
    background: #fff !important;
    border-color: #1a1a1a !important;
    border-bottom-color: #fff !important;
}

/* ─── DATAFRAME ─── */
[data-testid="stDataFrame"] {
    border: 2px solid #1a1a1a;
}

/* ─── BUTTON ─── */
.stButton > button {
    background: #1a1a1a !important;
    color: #F4F1EB !important;
    border: 2px solid #1a1a1a !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.15s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #D63F2E !important;
    border-color: #D63F2E !important;
}

/* Hide default Streamlit metric widget */
div[data-testid="stMetric"] { display: none; }

/* Fix warning text color in sidebar */
[data-testid="stSidebar"] .stAlert {
    background: #2a1a1a !important;
    border-color: #D97706 !important;
    color: #D97706 !important;
    font-size: 0.7rem !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MODEL SISTEM
# ============================================================================
class TahapanProyek:
    def __init__(self, nama, opt, ml, pes, faktor_risiko=None):
        self.nama          = nama
        self.optimistic    = opt
        self.most_likely   = ml
        self.pessimistic   = pes
        self.faktor_risiko = faktor_risiko or {}

        self.pert_mean = (opt + 4 * ml + pes) / 6
        span = pes - opt
        self.pert_std  = span / 6 if span > 0 else 0.01

    def jalankan(self, n):
        span = self.pessimistic - self.optimistic
        if span <= 0:
            dasar = np.full(n, float(self.optimistic))
        else:
            mu, sigma = self.pert_mean, self.pert_std
            denom     = max(sigma ** 2, 1e-9)
            d_lo      = max(mu - self.optimistic, 1e-9)
            d_hi      = max(self.pessimistic - mu, 1e-9)
            alpha = max((d_lo / span) * ((d_lo * d_hi) / denom - 1), 0.5)
            beta  = max(alpha * d_hi / d_lo, 0.5)
            dasar = self.optimistic + np.random.beta(alpha, beta, n) * span

        dampak_risiko = np.zeros(n)
        for _, info in self.faktor_risiko.items():
            terjadi        = np.random.random(n) < info['probability']
            dampak         = np.random.uniform(info['impact_min'], info['impact_max'], n)
            dampak_risiko += terjadi * dampak

        return np.maximum(dasar + dampak_risiko, self.optimistic * 0.8)


class SimulatorMC:
    def __init__(self, konfigurasi, n_sim, seed=42):
        np.random.seed(int(seed))
        self.n_sim   = n_sim
        self.tahapan = {}
        for nama, cfg in konfigurasi.items():
            bp = cfg['base_params']
            self.tahapan[nama] = TahapanProyek(
                nama          = nama,
                opt           = bp['optimistic'],
                ml            = bp['most_likely'],
                pes           = bp['pessimistic'],
                faktor_risiko = cfg.get('risk_factors', {})
            )

    def run(self):
        data = {nama: t.jalankan(self.n_sim) for nama, t in self.tahapan.items()}
        df          = pd.DataFrame(data)
        df['Total'] = df[list(self.tahapan.keys())].sum(axis=1)
        return df

    def jalur_kritis(self, df):
        total  = df['Total']
        thr    = float(np.percentile(total, 75))
        hasil  = {}
        for nama in self.tahapan:
            col     = df[nama]
            corr, _ = stats.spearmanr(col, total)
            mask    = col > col.median()
            n_mask  = int(mask.sum())
            prob    = float(np.mean(total[mask] > thr)) if n_mask > 0 else 0.0
            hasil[nama] = {
                'probability':  prob,
                'correlation':  float(corr) if not np.isnan(corr) else 0.0,
                'mean':         float(col.mean()),
                'contribution': float(col.mean() / total.mean() * 100)
            }
        return pd.DataFrame(hasil).T

    def kontribusi_risiko(self, df):
        jenis_risiko = [
            'cuaca_buruk', 'keterlambatan_material',
            'perubahan_desain', 'produktivitas_pekerja'
        ]
        contrib = {}
        for jr in jenis_risiko:
            probs, impacts, count = [], [], 0
            for t in self.tahapan.values():
                if jr in t.faktor_risiko:
                    info = t.faktor_risiko[jr]
                    probs.append(info['probability'])
                    impacts.append((info['impact_min'] + info['impact_max']) / 2)
                    count += 1
            if count:
                contrib[jr] = {
                    'avg_probability': float(np.mean(probs)),
                    'avg_impact':      float(np.mean(impacts)),
                    'stages_affected': count,
                    'risk_index':      float(np.mean(probs)) * float(np.mean(impacts)) * 100
                }
        return pd.DataFrame(contrib).T if contrib else pd.DataFrame()


# ============================================================================
# DATA KONFIGURASI DEFAULT
# ============================================================================
KONFIGURASI_TAHAPAN = {
    "Perencanaan_dan_Desain": {
        "base_params": {"optimistic": 1.5, "most_likely": 2.5, "pessimistic": 4.0},
        "risk_factors": {
            "perubahan_desain": {"probability": 0.45, "impact_min": 0.3, "impact_max": 1.2},
            "cuaca_buruk":      {"probability": 0.10, "impact_min": 0.1, "impact_max": 0.3},
        }
    },
    "Persiapan_Lahan_dan_Pondasi": {
        "base_params": {"optimistic": 1.0, "most_likely": 1.5, "pessimistic": 3.0},
        "risk_factors": {
            "cuaca_buruk":           {"probability": 0.50, "impact_min": 0.2, "impact_max": 0.8},
            "produktivitas_pekerja": {"probability": 0.30, "impact_min": 0.1, "impact_max": 0.5},
        }
    },
    "Struktur_Bangunan_5_Lantai": {
        "base_params": {"optimistic": 4.0, "most_likely": 6.0, "pessimistic": 9.0},
        "risk_factors": {
            "cuaca_buruk":            {"probability": 0.55, "impact_min": 0.5, "impact_max": 1.5},
            "keterlambatan_material": {"probability": 0.40, "impact_min": 0.3, "impact_max": 1.0},
            "produktivitas_pekerja":  {"probability": 0.35, "impact_min": 0.2, "impact_max": 0.8},
        }
    },
    "Instalasi_MEP": {
        "base_params": {"optimistic": 2.0, "most_likely": 3.0, "pessimistic": 5.0},
        "risk_factors": {
            "keterlambatan_material": {"probability": 0.45, "impact_min": 0.3, "impact_max": 1.2},
            "produktivitas_pekerja":  {"probability": 0.30, "impact_min": 0.1, "impact_max": 0.6},
        }
    },
    "Instalasi_Lab_Komputer_Elektro": {
        "base_params": {"optimistic": 1.0, "most_likely": 1.5, "pessimistic": 2.5},
        "risk_factors": {
            "keterlambatan_material": {"probability": 0.40, "impact_min": 0.2, "impact_max": 0.8},
            "perubahan_desain":       {"probability": 0.25, "impact_min": 0.1, "impact_max": 0.5},
        }
    },
    "Instalasi_Lab_VR_AR_Game": {
        "base_params": {"optimistic": 1.5, "most_likely": 2.5, "pessimistic": 4.5},
        "risk_factors": {
            "keterlambatan_material": {"probability": 0.65, "impact_min": 0.5, "impact_max": 2.0},
            "perubahan_desain":       {"probability": 0.40, "impact_min": 0.3, "impact_max": 1.0},
        }
    },
    "Instalasi_Lab_Mobile": {
        "base_params": {"optimistic": 0.5, "most_likely": 1.0, "pessimistic": 1.8},
        "risk_factors": {
            "keterlambatan_material": {"probability": 0.30, "impact_min": 0.1, "impact_max": 0.4},
        }
    },
    "Finishing_Interior_Furnitur": {
        "base_params": {"optimistic": 1.5, "most_likely": 2.5, "pessimistic": 4.0},
        "risk_factors": {
            "cuaca_buruk":            {"probability": 0.25, "impact_min": 0.1, "impact_max": 0.4},
            "keterlambatan_material": {"probability": 0.35, "impact_min": 0.2, "impact_max": 0.8},
            "produktivitas_pekerja":  {"probability": 0.30, "impact_min": 0.1, "impact_max": 0.5},
        }
    },
    "Pengujian_dan_Commissioning": {
        "base_params": {"optimistic": 0.5, "most_likely": 1.0, "pessimistic": 2.0},
        "risk_factors": {
            "perubahan_desain": {"probability": 0.35, "impact_min": 0.2, "impact_max": 0.8},
        }
    },
    "Administrasi_Serah_Terima": {
        "base_params": {"optimistic": 0.5, "most_likely": 0.8, "pessimistic": 1.5},
        "risk_factors": {
            "perubahan_desain": {"probability": 0.20, "impact_min": 0.1, "impact_max": 0.4},
        }
    },
}

LABEL_RISIKO = {
    'cuaca_buruk':            'Cuaca Buruk',
    'keterlambatan_material': 'Keterlambatan Material',
    'perubahan_desain':       'Perubahan Desain',
    'produktivitas_pekerja':  'Produktivitas Pekerja',
}


def layout_chart(**kw):
    """Return dict layout Plotly dengan tema terang-industrial."""
    grd = dict(gridcolor='rgba(0,0,0,0.07)', zerolinecolor='rgba(0,0,0,0.15)')
    bx  = {**grd, **kw.pop('xaxis', {})}
    by  = {**grd, **kw.pop('yaxis', {})}
    base = dict(
        paper_bgcolor='rgba(255,255,255,0)',
        plot_bgcolor='rgba(255,255,255,0.9)',
        font=dict(family='IBM Plex Mono', color='#1a1a1a', size=11),
        xaxis=bx, yaxis=by,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    base.update(kw)
    return base


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### ⚙ Simulasi")
    n_sim = st.slider("Jumlah Iterasi", 5_000, 50_000, 20_000, 1_000)
    seed  = st.number_input("Seed Acak", min_value=0, max_value=9999, value=42, step=1)

    st.markdown("### 📋 Durasi Tahapan")
    stages_cfg = {}
    for stage_name, cfg in KONFIGURASI_TAHAPAN.items():
        label = stage_name.replace('_', ' ')
        with st.expander(label, expanded=False):
            bp  = cfg['base_params']
            opt = st.number_input(
                "Optimistik (bln)", min_value=0.1, max_value=24.0,
                value=float(bp['optimistic']), step=0.1, key=f"o_{stage_name}"
            )
            ml  = st.number_input(
                "Most Likely (bln)", min_value=0.1, max_value=24.0,
                value=float(bp['most_likely']), step=0.1, key=f"m_{stage_name}"
            )
            pes = st.number_input(
                "Pesimistik (bln)", min_value=0.1, max_value=36.0,
                value=float(bp['pessimistic']), step=0.1, key=f"p_{stage_name}"
            )
            # Validasi urutan nilai
            if opt > ml:
                st.warning("⚠ Optimistik harus ≤ Most Likely")
                ml = opt
            if ml > pes:
                st.warning("⚠ Most Likely harus ≤ Pesimistik")
                pes = ml + 0.1

            stages_cfg[stage_name] = {
                'base_params':  {'optimistic': opt, 'most_likely': ml, 'pessimistic': pes},
                'risk_factors': cfg.get('risk_factors', {})
            }

    st.markdown("---")
    run_btn = st.button("▶ Jalankan Simulasi", use_container_width=True)
    st.markdown(
        "<div style='font-size:0.65rem;color:#555;margin-top:1rem;"
        "font-family:IBM Plex Mono,monospace;line-height:1.8;'>"
        "MODSIM · Praktikum 5<br>[11S1221] Monte Carlo<br>Gedung FITE 5 Lantai</div>",
        unsafe_allow_html=True
    )


# ============================================================================
# HEADER UTAMA
# ============================================================================
st.markdown("""
<div class="top-header">
  <div class="top-header-left">
    <div class="label">MODSIM · Praktikum 5 · [11S1221] · Monte Carlo Simulation</div>
    <h1>Estimasi Waktu Konstruksi<br>Gedung FITE — 5 Lantai</h1>
  </div>
  <div class="top-header-right">
    Simulasi Probabilistik<br>
    Distribusi PERT + Beta<br>
    Faktor Risiko Aktif<br>
    Analisis Jalur Kritis
  </div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE & SIMULASI
# ============================================================================
if 'df' not in st.session_state:
    st.session_state.df  = None
    st.session_state.sim = None

if run_btn:
    with st.spinner("Menjalankan simulasi..."):
        try:
            sim = SimulatorMC(stages_cfg, int(n_sim), int(seed))
            df  = sim.run()
            st.session_state.df  = df
            st.session_state.sim = sim
            st.success(f"✅ Selesai — {int(n_sim):,} iterasi berhasil dijalankan.")
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

# ── Belum dijalankan ─────────────────────────────────────────────────────────
if st.session_state.df is None:
    st.markdown("""
    <div style="border:2px solid #1a1a1a;background:#fff;padding:3rem;
                text-align:center;margin-top:1rem;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                    letter-spacing:0.15em;text-transform:uppercase;
                    color:#888;margin-bottom:0.8rem;">Status Simulasi</div>
        <div style="font-size:1.4rem;font-weight:700;color:#1a1a1a;margin-bottom:0.5rem;">
            Menunggu Input
        </div>
        <div style="font-size:0.85rem;color:#888;">
            Sesuaikan parameter di sidebar, lalu klik
            <b>▶ Jalankan Simulasi</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="sec-title">Konfigurasi Awal Tahapan Proyek</div>',
        unsafe_allow_html=True
    )
    rows = []
    for name, cfg in KONFIGURASI_TAHAPAN.items():
        bp = cfg['base_params']
        rows.append({
            'Tahapan':           name.replace('_', ' '),
            'Optimistik (bln)':  bp['optimistic'],
            'Most Likely (bln)': bp['most_likely'],
            'Pesimistik (bln)':  bp['pessimistic'],
            'PERT Mean': round(
                (bp['optimistic'] + 4 * bp['most_likely'] + bp['pessimistic']) / 6, 2
            )
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.stop()


# ── Ada hasil ────────────────────────────────────────────────────────────────
df          = st.session_state.df
sim         = st.session_state.sim
total       = df['Total']
stage_names = list(sim.tahapan.keys())
total_arr   = total.values

mean_t = float(total.mean())
med_t  = float(total.median())
std_t  = float(total.std())
ci_80  = np.percentile(total_arr, [10, 90])
ci_95  = np.percentile(total_arr, [2.5, 97.5])
p16    = float(np.mean(total_arr <= 16) * 100)
p20    = float(np.mean(total_arr <= 20) * 100)
p24    = float(np.mean(total_arr <= 24) * 100)


# ============================================================================
# METRIC ROW
# ============================================================================
st.markdown(f"""
<div class="metric-row">
  <div class="metric-box accent-red">
    <div class="m-label">Rata-rata Durasi</div>
    <div class="m-val">{mean_t:.1f}</div>
    <div class="m-unit">bulan</div>
  </div>
  <div class="metric-box accent-blue">
    <div class="m-label">Median</div>
    <div class="m-val">{med_t:.1f}</div>
    <div class="m-unit">bulan</div>
  </div>
  <div class="metric-box accent-org">
    <div class="m-label">Std Deviasi</div>
    <div class="m-val">{std_t:.1f}</div>
    <div class="m-unit">bulan</div>
  </div>
  <div class="metric-box accent-prp">
    <div class="m-label">80% CI</div>
    <div class="m-val" style="font-size:1.1rem;">{ci_80[0]:.1f}–{ci_80[1]:.1f}</div>
    <div class="m-unit">bulan</div>
  </div>
  <div class="metric-box accent-grn">
    <div class="m-label">95% CI</div>
    <div class="m-val" style="font-size:1.1rem;">{ci_95[0]:.1f}–{ci_95[1]:.1f}</div>
    <div class="m-unit">bulan</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "DISTRIBUSI & DEADLINE",
    "CRITICAL PATH",
    "ANALISIS RISIKO",
    "RESOURCE",
    "LAPORAN LENGKAP"
])


# ─── TAB 1 ───────────────────────────────────────────────────────────────────
with tab1:
    ca, cb = st.columns(2)

    with ca:
        st.markdown(
            '<div class="sec-title">Distribusi Durasi Total Proyek</div>',
            unsafe_allow_html=True
        )
        kde_x  = np.linspace(float(total_arr.min()), float(total_arr.max()), 400)
        kde_fn = stats.gaussian_kde(total_arr)

        fig1 = go.Figure()
        fig1.add_trace(go.Histogram(
            x=total_arr, nbinsx=60, histnorm='probability density',
            name='Histogram', marker_color='rgba(37,99,235,0.3)',
            marker_line_color='rgba(37,99,235,0.8)', marker_line_width=0.5
        ))
        fig1.add_trace(go.Scatter(
            x=kde_x, y=kde_fn(kde_x), mode='lines', name='KDE',
            line=dict(color='#2563EB', width=2.5)
        ))
        for span, fc, lbl in [
            ([float(ci_95[0]), float(ci_95[1])], 'rgba(214,63,46,0.07)', '95% CI'),
            ([float(ci_80[0]), float(ci_80[1])], 'rgba(217,119,6,0.12)', '80% CI'),
        ]:
            fig1.add_vrect(
                x0=span[0], x1=span[1], fillcolor=fc, line_width=0,
                annotation_text=lbl, annotation_font_size=9,
                annotation_font_color='#555'
            )
        fig1.add_vline(
            x=mean_t, line_dash='dash', line_color='#D63F2E',
            annotation_text=f'Mean {mean_t:.1f}', annotation_font_size=9
        )
        fig1.add_vline(
            x=med_t, line_dash='dot', line_color='#059669',
            annotation_text=f'Med {med_t:.1f}', annotation_font_size=9
        )
        for dl, col, lbl in [(16,'#D63F2E','16bln'),(20,'#D97706','20bln'),(24,'#059669','24bln')]:
            p = float(np.mean(total_arr <= dl) * 100)
            fig1.add_vline(
                x=dl, line_color=col, line_width=1.2, line_dash='dashdot',
                annotation_text=f'{lbl} ({p:.0f}%)', annotation_font_size=8
            )
        fig1.update_layout(**layout_chart(
            title=dict(text='Histogram + KDE', font=dict(size=12), x=0),
            xaxis_title='Durasi (Bulan)', yaxis_title='Densitas',
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=9)),
            height=370
        ))
        st.plotly_chart(fig1, use_container_width=True)

    with cb:
        st.markdown(
            '<div class="sec-title">Kurva CDF Penyelesaian</div>',
            unsafe_allow_html=True
        )
        dls   = np.arange(8, 42, 0.5)
        probs = [float(np.mean(total_arr <= d)) for d in dls]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=dls, y=probs, mode='lines', name='P(Selesai)',
            line=dict(color='#7C3AED', width=3),
            fill='tozeroy', fillcolor='rgba(124,58,237,0.08)'
        ))
        for ref, col, lbl in [(0.5,'#D63F2E','50%'),(0.80,'#059669','80%'),(0.95,'#2563EB','95%')]:
            fig2.add_hline(
                y=ref, line_dash='dash', line_color=col,
                annotation_text=lbl, annotation_position='right',
                annotation_font_size=9
            )
        for dl, col in [(16,'#D63F2E'),(20,'#D97706'),(24,'#059669')]:
            p_val = float(np.mean(total_arr <= dl))
            fig2.add_trace(go.Scatter(
                x=[dl], y=[p_val], mode='markers+text',
                marker=dict(size=10, color=col, symbol='diamond',
                            line=dict(color='white', width=1.5)),
                text=[f'{p_val:.0%}'], textposition='top center',
                textfont=dict(size=9, color=col), showlegend=False
            ))
        fig2.update_layout(**layout_chart(
            title=dict(text='Kurva CDF', font=dict(size=12), x=0),
            xaxis_title='Deadline (Bulan)', yaxis_title='Probabilitas',
            yaxis=dict(tickformat='.0%', range=[-0.03, 1.08],
                       gridcolor='rgba(0,0,0,0.07)'),
            height=370
        ))
        st.plotly_chart(fig2, use_container_width=True)

    # Deadline tiles
    st.markdown(
        '<div class="sec-title">Probabilitas per Deadline</div>',
        unsafe_allow_html=True
    )
    tiles_html = '<div class="dl-grid">'
    for dl in [12, 14, 16, 18, 20, 22, 24]:
        p   = float(np.mean(total_arr <= dl) * 100)
        cls = 'dl-ok' if p >= 70 else ('dl-warn' if p >= 35 else 'dl-bad')
        tiles_html += (
            f'<div class="dl-tile {cls}">'
            f'<div class="dl-num">{dl} BLN</div>'
            f'<div class="dl-pct">{p:.1f}%</div>'
            f'</div>'
        )
    tiles_html += '</div>'
    st.markdown(tiles_html, unsafe_allow_html=True)

    # Violin plot per tahapan
    st.markdown(
        '<div class="sec-title">Sebaran Durasi per Tahapan</div>',
        unsafe_allow_html=True
    )
    palette = ['#2563EB','#D63F2E','#059669','#D97706','#7C3AED',
               '#0891B2','#DB2777','#65A30D','#EA580C','#4F46E5']
    fig_vio = go.Figure()
    for i, s in enumerate(stage_names):
        hex_c = palette[i % len(palette)]
        r = int(hex_c[1:3], 16)
        g = int(hex_c[3:5], 16)
        b = int(hex_c[5:7], 16)
        fig_vio.add_trace(go.Violin(
            y=df[s].values,
            name=s.replace('_', ' '),
            box_visible=True,
            meanline_visible=True,
            fillcolor=f'rgba({r},{g},{b},0.25)',
            line_color=hex_c,
            points='outliers',
            marker=dict(size=2, opacity=0.3)
        ))
    fig_vio.update_layout(**layout_chart(
        title=dict(text='Violin Plot Durasi Tiap Tahapan', font=dict(size=12), x=0),
        yaxis_title='Durasi (Bulan)',
        showlegend=False,
        height=420,
        xaxis=dict(tickfont=dict(size=8))
    ))
    st.plotly_chart(fig_vio, use_container_width=True)


# ─── TAB 2 ───────────────────────────────────────────────────────────────────
with tab2:
    cp = sim.jalur_kritis(df).sort_values('probability', ascending=True)
    st.markdown(
        '<div class="sec-title">Analisis Jalur Kritis</div>',
        unsafe_allow_html=True
    )

    colors_cp = [
        '#D63F2E' if p > 0.55 else ('#D97706' if p > 0.38 else '#059669')
        for p in cp['probability']
    ]
    fig_cp = go.Figure()
    fig_cp.add_trace(go.Bar(
        y=[n.replace('_', ' ') for n in cp.index],
        x=cp['probability'].values,
        orientation='h',
        marker_color=colors_cp,
        marker_line_color='#1a1a1a',
        marker_line_width=1,
        text=[f'{p:.1%}' for p in cp['probability']],
        textposition='outside',
        textfont=dict(size=10, color='#1a1a1a')
    ))
    fig_cp.add_vline(
        x=0.55, line_dash='dash', line_color='#D63F2E',
        annotation_text='Sangat Kritis', annotation_font_size=9
    )
    fig_cp.add_vline(
        x=0.38, line_dash='dash', line_color='#D97706',
        annotation_text='Kritis', annotation_font_size=9
    )
    fig_cp.update_layout(**layout_chart(
        title=dict(text='Probabilitas Menjadi Critical Path', font=dict(size=12), x=0),
        xaxis_title='Probabilitas',
        xaxis=dict(tickformat='.0%', range=[0, 1.15], gridcolor='rgba(0,0,0,0.07)'),
        yaxis=dict(tickfont=dict(size=8)),
        height=460
    ))
    st.plotly_chart(fig_cp, use_container_width=True)

    top3  = cp.sort_values('probability', ascending=False).head(3)
    c1c, c2c, c3c = st.columns(3)
    labels_rank   = ['#1 Paling Kritis', '#2 Kritis', '#3 Kritis']
    for i, (col_w, (name, row)) in enumerate(zip([c1c, c2c, c3c], top3.iterrows())):
        cls = (
            'chip-bad' if row['probability'] > 0.55
            else ('chip-warn' if row['probability'] > 0.38 else 'chip-ok')
        )
        with col_w:
            st.markdown(f"""
            <div class="crit-card">
              <div class="rank">{labels_rank[i]}</div>
              <div class="name">{name.replace('_', ' ')}</div>
              <span class="chip {cls}">{row['probability']:.1%}</span><br>
              <small style="color:#888;font-size:0.7rem;
                            font-family:'IBM Plex Mono',monospace;">
                Kontribusi {row['contribution']:.1f}% &nbsp;|&nbsp;
                r={row['correlation']:.3f}
              </small>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(
        '<div class="sec-title">Tabel Detail</div>',
        unsafe_allow_html=True
    )
    cp_show = cp.sort_values('probability', ascending=False).copy()
    cp_show['Status'] = cp_show['probability'].apply(
        lambda p: '🔴 Sangat Kritis' if p > 0.55 else ('🟠 Kritis' if p > 0.38 else '🟢 Normal')
    )
    cp_show         = cp_show[['probability','correlation','mean','contribution','Status']]
    cp_show.columns = ['P. Kritis','Korelasi','Mean (bln)','Kontribusi (%)','Status']
    cp_show.index   = [n.replace('_', ' ') for n in cp_show.index]
    st.dataframe(cp_show.round(3), use_container_width=True)


# ─── TAB 3 ───────────────────────────────────────────────────────────────────
with tab3:
    rc = sim.kontribusi_risiko(df)
    cr1, cr2 = st.columns(2)

    with cr1:
        st.markdown(
            '<div class="sec-title">Risk Index per Faktor</div>',
            unsafe_allow_html=True
        )
        if not rc.empty:
            rc_sorted = rc.sort_values('risk_index', ascending=False)
            clrs_r    = ['#D63F2E','#D97706','#D97706','#059669']
            fig_r     = go.Figure()
            fig_r.add_trace(go.Bar(
                x=[LABEL_RISIKO.get(n, n) for n in rc_sorted.index],
                y=rc_sorted['risk_index'].values,
                marker_color=clrs_r[:len(rc_sorted)],
                marker_line_color='#1a1a1a', marker_line_width=1,
                text=[f'{v:.1f}' for v in rc_sorted['risk_index']],
                textposition='outside',
                textfont=dict(size=11, color='#1a1a1a')
            ))
            fig_r.update_layout(**layout_chart(
                title=dict(text='Risk Index', font=dict(size=12), x=0),
                yaxis_title='Risk Index',
                height=320
            ))
            st.plotly_chart(fig_r, use_container_width=True)

    with cr2:
        st.markdown(
            '<div class="sec-title">Matriks Korelasi Tahapan</div>',
            unsafe_allow_html=True
        )
        corr_m = df[stage_names].corr()
        labels = [n.replace('_', ' ') for n in corr_m.columns]
        fig_hm = go.Figure(go.Heatmap(
            z=corr_m.values, x=labels, y=labels,
            colorscale='RdBu', zmid=0,
            text=np.round(corr_m.values, 2),
            texttemplate='%{text}',
            textfont=dict(size=7, color='#1a1a1a'),
            colorbar=dict(tickfont=dict(color='#1a1a1a'))
        ))
        fig_hm.update_layout(**layout_chart(
            title=dict(text='Korelasi Antar Tahapan', font=dict(size=12), x=0),
            xaxis=dict(tickfont=dict(size=7)),
            yaxis=dict(tickfont=dict(size=7)),
            height=320
        ))
        st.plotly_chart(fig_hm, use_container_width=True)

    if not rc.empty:
        st.markdown(
            '<div class="sec-title">Detail Faktor Risiko</div>',
            unsafe_allow_html=True
        )
        rc_show         = rc.copy()
        rc_show.index   = [LABEL_RISIKO.get(n, n) for n in rc_show.index]
        rc_show.columns = ['Avg Probabilitas','Avg Dampak (bln)','Tahapan Terdampak','Risk Index']
        st.dataframe(rc_show.round(3), use_container_width=True)


# ─── TAB 4 ───────────────────────────────────────────────────────────────────
with tab4:
    st.markdown(
        '<div class="sec-title">Analisis Penambahan Resource</div>',
        unsafe_allow_html=True
    )
    NILAI_WAKTU = 150_000_000
    SKENARIO = [
        {'stage':'Struktur_Bangunan_5_Lantai',   'resource':'Alat Berat Tambahan',
         'qty':2,  'eff':0.20, 'dur':6,   'biaya_satuan':25_000_000},
        {'stage':'Struktur_Bangunan_5_Lantai',   'resource':'Pekerja Khusus',
         'qty':10, 'eff':0.15, 'dur':6,   'biaya_satuan':8_000_000},
        {'stage':'Instalasi_Lab_VR_AR_Game',     'resource':'Insinyur Spesialis',
         'qty':3,  'eff':0.25, 'dur':3,   'biaya_satuan':20_000_000},
        {'stage':'Instalasi_MEP',                'resource':'Insinyur MEP',
         'qty':2,  'eff':0.22, 'dur':3,   'biaya_satuan':20_000_000},
        {'stage':'Finishing_Interior_Furnitur',  'resource':'Pekerja Finishing',
         'qty':8,  'eff':0.18, 'dur':2.5, 'biaya_satuan':8_000_000},
    ]

    sc_results = []
    for sc in SKENARIO:
        rn     = df.copy()
        factor = min(sc['eff'] * sc['qty'], 0.55)
        if sc['stage'] in rn.columns:
            rn[sc['stage']] = rn[sc['stage']] * (1 - factor)
        rn['Total'] = rn[stage_names].sum(axis=1)
        saved   = float(total.mean() - rn['Total'].mean())
        cost    = sc['biaya_satuan'] * sc['qty'] * sc['dur']
        benefit = saved * NILAI_WAKTU
        roi     = (benefit - cost) / cost * 100 if cost > 0 else 0.0
        sc_results.append({**sc, 'saved': saved, 'cost': cost, 'roi': roi, 'rn': rn})

    rs1, rs2 = st.columns(2)
    with rs1:
        labels_sc = [f"S{i+1}: {s['resource']}" for i, s in enumerate(SKENARIO)]
        fig_sv    = go.Figure(go.Bar(
            y=labels_sc,
            x=[s['saved'] for s in sc_results],
            orientation='h',
            marker_color='rgba(5,150,105,0.6)',
            marker_line_color='#059669', marker_line_width=1.2,
            text=[f"{s['saved']:.2f} bln" for s in sc_results],
            textposition='outside',
            textfont=dict(color='#1a1a1a', size=9)
        ))
        fig_sv.update_layout(**layout_chart(
            title=dict(text='Penghematan Durasi', font=dict(size=12), x=0),
            xaxis_title='Bulan',
            yaxis=dict(tickfont=dict(size=8)),
            height=300
        ))
        st.plotly_chart(fig_sv, use_container_width=True)

    with rs2:
        roi_v   = [s['roi'] for s in sc_results]
        rclrs   = ['#059669' if r > 0 else '#D63F2E' for r in roi_v]
        fig_roi = go.Figure(go.Bar(
            x=[f'S{i+1}' for i in range(len(SKENARIO))],
            y=roi_v,
            marker_color=rclrs,
            marker_line_color='#1a1a1a', marker_line_width=1,
            text=[f'{r:.0f}%' for r in roi_v],
            textposition='outside',
            textfont=dict(color='#1a1a1a', size=10)
        ))
        fig_roi.add_hline(y=0, line_color='#1a1a1a', line_width=1)
        fig_roi.update_layout(**layout_chart(
            title=dict(text='Return on Investment', font=dict(size=12), x=0),
            yaxis_title='ROI (%)',
            height=300
        ))
        st.plotly_chart(fig_roi, use_container_width=True)

    # Cost-Benefit
    fig_cb = go.Figure()
    fig_cb.add_trace(go.Bar(
        name='Biaya (Jt Rp)',
        x=[f'S{i+1}' for i in range(len(SKENARIO))],
        y=[s['cost'] / 1e6 for s in sc_results],
        marker_color='rgba(217,119,6,0.6)',
        marker_line_color='#D97706', marker_line_width=1
    ))
    fig_cb.add_trace(go.Bar(
        name='Benefit (Jt Rp)',
        x=[f'S{i+1}' for i in range(len(SKENARIO))],
        y=[s['saved'] * NILAI_WAKTU / 1e6 for s in sc_results],
        marker_color='rgba(37,99,235,0.6)',
        marker_line_color='#2563EB', marker_line_width=1
    ))
    fig_cb.update_layout(**layout_chart(
        title=dict(text='Cost–Benefit per Skenario', font=dict(size=12), x=0),
        barmode='group', yaxis_title='Juta Rupiah',
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=9)),
        height=300
    ))
    st.plotly_chart(fig_cb, use_container_width=True)

    # Perbandingan deadline
    st.markdown(
        '<div class="sec-title">Peningkatan Probabilitas Deadline</div>',
        unsafe_allow_html=True
    )
    dls_cmp = [16, 20, 24]
    fig_dl  = go.Figure()
    base_p  = [float(np.mean(total_arr <= d) * 100) for d in dls_cmp]
    fig_dl.add_trace(go.Bar(
        name='Baseline',
        x=[f'{d} bln' for d in dls_cmp],
        y=base_p,
        marker_color='rgba(100,116,139,0.5)',
        marker_line_color='#64748b', marker_line_width=1
    ))
    clrs5 = ['#2563EB','#059669','#D97706','#7C3AED','#D63F2E']
    for i, sc in enumerate(sc_results):
        d_col = sc['rn']['Total'].values
        sc_p  = [float(np.mean(d_col <= d) * 100) for d in dls_cmp]
        fig_dl.add_trace(go.Bar(
            name=f"S{i+1}: {sc['resource']}",
            x=[f'{d} bln' for d in dls_cmp],
            y=sc_p,
            marker_color=clrs5[i],
            marker_line_color='#1a1a1a', marker_line_width=0.5
        ))
    fig_dl.update_layout(**layout_chart(
        title=dict(text='Baseline vs Skenario Resource', font=dict(size=12), x=0),
        barmode='group', yaxis_title='Probabilitas (%)',
        yaxis=dict(range=[0, 110]),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=9)),
        height=360
    ))
    st.plotly_chart(fig_dl, use_container_width=True)

    st.markdown(
        '<div class="sec-title">Ringkasan Skenario</div>',
        unsafe_allow_html=True
    )
    tbl = []
    for i, sc in enumerate(sc_results):
        tbl.append({
            'No':           f'S{i+1}',
            'Tahapan':      sc['stage'].replace('_', ' '),
            'Resource':     sc['resource'],
            'Qty':          sc['qty'],
            'Durasi (bln)': sc['dur'],
            'Hemat (bln)':  round(sc['saved'], 2),
            'Biaya (Jt)':   round(sc['cost'] / 1e6, 1),
            'ROI (%)':      round(sc['roi'], 1)
        })
    st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)


# ─── TAB 5 ───────────────────────────────────────────────────────────────────
with tab5:
    st.markdown(
        '<div class="sec-title">Laporan Statistik Lengkap</div>',
        unsafe_allow_html=True
    )

    # Jawaban 1
    with st.expander("Jawaban 1 — Total Waktu Pembangunan", expanded=True):
        j1a, j1b = st.columns(2)
        with j1a:
            st.markdown(f"""
            <div class="info-panel">
              <div class="ip-title">Statistik Deskriptif</div>
              <b>Rata-rata :</b> {mean_t:.2f} bulan<br>
              <b>Median    :</b> {med_t:.2f} bulan<br>
              <b>Std Dev   :</b> {std_t:.2f} bulan<br>
              <b>Minimum   :</b> {float(total_arr.min()):.2f} bulan<br>
              <b>Maximum   :</b> {float(total_arr.max()):.2f} bulan<br>
              <b>Skewness  :</b> {float(stats.skew(total_arr)):.3f}<br>
              <b>Kurtosis  :</b> {float(stats.kurtosis(total_arr)):.3f}
            </div>
            """, unsafe_allow_html=True)
        with j1b:
            p5  = float(np.percentile(total_arr, 5))
            p95 = float(np.percentile(total_arr, 95))
            st.markdown(f"""
            <div class="info-panel">
              <div class="ip-title">Confidence Intervals</div>
              <b>80% CI :</b> [{ci_80[0]:.2f}, {ci_80[1]:.2f}] bulan<br>
              <b>90% CI :</b> [{p5:.2f}, {p95:.2f}] bulan<br>
              <b>95% CI :</b> [{ci_95[0]:.2f}, {ci_95[1]:.2f}] bulan<br><br>
              <b>Kesimpulan:</b><br>
              Proyek selesai dalam <b>{mean_t:.1f} ± {std_t:.1f} bulan</b>.<br>
              Rentang 80% skenario: <b>{ci_80[0]:.1f}–{ci_80[1]:.1f} bulan</b>.
            </div>
            """, unsafe_allow_html=True)

        rows_j1 = []
        for s in stage_names:
            col_s = df[s].values
            rows_j1.append({
                'Tahapan': s.replace('_', ' '),
                'Mean':    round(float(col_s.mean()), 2),
                'Std':     round(float(col_s.std()), 2),
                'Min':     round(float(col_s.min()), 2),
                'Max':     round(float(col_s.max()), 2)
            })
        rows_j1.append({
            'Tahapan': 'TOTAL',
            'Mean':    round(mean_t, 2),
            'Std':     round(std_t, 2),
            'Min':     round(float(total_arr.min()), 2),
            'Max':     round(float(total_arr.max()), 2)
        })
        st.dataframe(pd.DataFrame(rows_j1), use_container_width=True, hide_index=True)

    # Jawaban 2
    with st.expander("Jawaban 2 — Risiko Keterlambatan", expanded=False):
        baseline = sum(t.pert_mean for t in sim.tahapan.values())
        added    = mean_t - baseline
        p90_val  = float(np.percentile(total_arr, 90))
        st.markdown(f"""
        <div class="info-panel">
          <div class="ip-title">Dampak Risiko</div>
          <b>Baseline PERT (tanpa risiko) :</b> {baseline:.2f} bulan<br>
          <b>Simulasi (dengan risiko)      :</b> {mean_t:.2f} bulan<br>
          <b>Penambahan akibat risiko      :</b> +{added:.2f} bulan
            ({added/baseline*100:.1f}%)<br>
          <b>Value at Risk 90% (P90)       :</b> {p90_val:.2f} bulan
        </div>
        """, unsafe_allow_html=True)
        if not rc.empty:
            rc2         = rc.copy()
            rc2.index   = [LABEL_RISIKO.get(n, n) for n in rc2.index]
            rc2.columns = ['Avg Prob','Avg Dampak (bln)','Thn Terdampak','Risk Index']
            st.dataframe(rc2.round(3), use_container_width=True)

    # Jawaban 3
    with st.expander("Jawaban 3 — Jalur Kritis", expanded=False):
        cp3         = sim.jalur_kritis(df).sort_values('probability', ascending=False)
        cp3['Status'] = cp3['probability'].apply(
            lambda p: '🔴 Sangat Kritis' if p > 0.55
            else ('🟠 Kritis' if p > 0.38 else '🟢 Normal')
        )
        cp3.index   = [n.replace('_', ' ') for n in cp3.index]
        cp3.columns = ['P. Kritis','Korelasi','Mean (bln)','Kontribusi (%)','Status']
        st.dataframe(cp3.round(3), use_container_width=True)
        top3n = cp3.index[:3].tolist()
        st.markdown(f"""
        <div class="rec-panel">
          <b>3 Tahapan Paling Kritis:</b><br>
          1. {top3n[0]} — {cp3.iloc[0]['P. Kritis']:.1%}<br>
          2. {top3n[1]} — {cp3.iloc[1]['P. Kritis']:.1%}<br>
          3. {top3n[2]} — {cp3.iloc[2]['P. Kritis']:.1%}
        </div>
        """, unsafe_allow_html=True)

    # Jawaban 4
    with st.expander("Jawaban 4 — Probabilitas Deadline", expanded=False):
        rows4 = []
        for dl in [14, 16, 18, 20, 22, 24]:
            p_on     = float(np.mean(total_arr <= dl))
            p_lt     = 1.0 - p_on
            late_arr = total_arr[total_arr > dl]
            late     = float(late_arr.mean() - dl) if len(late_arr) > 0 else 0.0
            flag     = '✅' if p_on >= 0.80 else ('⚠️' if p_on >= 0.40 else '❌')
            rows4.append({
                'DL (bln)':        dl,
                'P(Tepat)':        f'{p_on:.1%}',
                'P(Terlambat)':    f'{p_lt:.1%}',
                'Rata-rata Telat': f'{late:.2f} bln',
                'Status':          flag
            })
        st.dataframe(pd.DataFrame(rows4), use_container_width=True, hide_index=True)
        p50_v = float(np.percentile(total_arr, 50))
        p80_v = float(np.percentile(total_arr, 80))
        p90_v = float(np.percentile(total_arr, 90))
        p95_v = float(np.percentile(total_arr, 95))
        st.markdown(f"""
        <div class="info-panel">
          <div class="ip-title">Deadline per Tingkat Kepercayaan</div>
          50% confidence → {p50_v:.1f} bulan<br>
          80% confidence → {p80_v:.1f} bulan<br>
          90% confidence → {p90_v:.1f} bulan<br>
          95% confidence → {p95_v:.1f} bulan<br><br>
          <b>Deadline 16 bln:</b> {p16:.1f}% →
            {'⚠️ SANGAT BERISIKO' if p16 < 30 else '🟠 BERISIKO'}<br>
          <b>Deadline 20 bln:</b> {p20:.1f}% →
            {'✅ REALISTIS' if p20 >= 50 else '⚠️ BERISIKO'}<br>
          <b>Deadline 24 bln:</b> {p24:.1f}% →
            {'✅ SANGAT AMAN' if p24 >= 80 else '🟠 REALISTIS'}
        </div>
        """, unsafe_allow_html=True)

    # Jawaban 5
    with st.expander("Jawaban 5 — Pengaruh Penambahan Resource", expanded=False):
        rows5 = []
        for i, sc in enumerate(sc_results):
            rows5.append({
                'No':           f'S{i+1}',
                'Tahapan':      sc['stage'].replace('_', ' '),
                'Resource':     sc['resource'],
                'Qty':          sc['qty'],
                'Hemat (bln)':  round(sc['saved'], 2),
                'Biaya (Jt)':   round(sc['cost'] / 1e6, 1),
                'ROI (%)':      round(sc['roi'], 1),
                'Status':       '✅' if sc['roi'] > 0 else '❌'
            })
        st.dataframe(pd.DataFrame(rows5), use_container_width=True, hide_index=True)
        best_roi   = max(sc_results, key=lambda x: x['roi'])
        most_saved = max(sc_results, key=lambda x: x['saved'])
        p80_v = float(np.percentile(total_arr, 80))
        p95_v = float(np.percentile(total_arr, 95))
        sb    = p80_v - mean_t
        cr    = p95_v - mean_t
        st.markdown(f"""
        <div class="rec-panel">
          <b>ROI Terbaik:</b> {best_roi['resource']}
            di {best_roi['stage'].replace('_', ' ')} → ROI {best_roi['roi']:.1f}%<br>
          <b>Hemat Terbesar:</b> {most_saved['resource']}
            → {most_saved['saved']:.2f} bulan<br><br>
          <b>Rekomendasi Manajemen Risiko:</b><br>
          · Safety Buffer (80% conf): +{sb:.2f} bulan<br>
          · Contingency Reserve (95%): +{cr:.2f} bulan<br>
          · Jadwal rekomendasi: {mean_t:.1f} + {sb:.1f} =
            <b>{mean_t + sb:.1f} bulan</b>
        </div>
        """, unsafe_allow_html=True)

    # Cek deadline kustom
    st.markdown(
        '<div class="sec-title">Cek Deadline Target</div>',
        unsafe_allow_html=True
    )
    target   = st.slider("Deadline target (bulan):", 10, 36, 20, 1)
    p_target = float(np.mean(total_arr <= target))
    d_risk   = max(0.0, float(np.percentile(total_arr, 95)) - target)
    cls_t    = (
        'chip-ok'   if p_target >= 0.70
        else ('chip-warn' if p_target >= 0.35 else 'chip-bad')
    )
    st.markdown(f"""
    <div class="info-panel">
      Deadline <b>{target} bulan</b> &nbsp;→&nbsp;
      <span class="chip {cls_t}">{p_target:.1%} peluang selesai tepat waktu</span><br>
      <small style="color:#888;font-family:'IBM Plex Mono',monospace;font-size:0.7rem;">
        Potensi keterlambatan P95: {d_risk:.2f} bulan
      </small>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
     padding:0.8rem 0;font-family:'IBM Plex Mono',monospace;
     font-size:0.65rem;color:#888;">
  <span>MODSIM · Praktikum 5 · [11S1221] · Gedung FITE 5 Lantai</span>
  <span>{int(n_sim):,} iterasi &nbsp;|&nbsp; Seed: {int(seed)}</span>
  <span>Hasil bersifat estimasi probabilistik</span>
</div>
""", unsafe_allow_html=True)