"""
app.py — Auto EDA Dashboard (Bahasa Indonesia)
────────────────────────────────────────────────────────────────
Dashboard EDA otomatis senior-grade.
Menerima CSV / XLSX apapun. Tanpa asumsi nama kolom.
Tab: Ringkasan · Data · Statistik · Missing · Outlier ·
     Korelasi · Visualisasi · Keputusan · AI Insight · Ekspor
────────────────────────────────────────────────────────────────
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Auto EDA Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from modules.data_loader     import DataLoader
from modules.eda_analysis    import EDAAnalyzer
from modules.visualizations  import Visualizer
from modules.ai_insights     import AIInsights
from modules.export_utils    import ExportUtils
from modules.decision_theory import DecisionTheoryAnalyzer
from modules.decision_viz    import DecisionVisualizer

# ─────────────────────────────────────────────────────────────
# CSS GLOBAL
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
:root{
  --bg:      #050d1a; --card:   #0a1628; --card2:  #0f1f35;
  --border:  #1a2f50; --cyan:   #00d4ff; --cyan2:  #00a3c4;
  --violet:  #7c3aed; --green:  #10b981; --amber:  #f59e0b;
  --coral:   #ff6b6b; --txt:    #e2e8f0; --muted:  #64748b;
  --dim:     #334155;
}
.stApp,[data-testid="stAppViewContainer"]{background:var(--bg)!important;font-family:'IBM Plex Sans',sans-serif;}
[data-testid="stSidebar"]{background:var(--card)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{color:var(--txt)!important;}
#MainMenu,footer,header{visibility:hidden;}

/* Kartu metrik */
.mc{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px 22px;text-align:center;position:relative;overflow:hidden;transition:border-color .25s,transform .2s;}
.mc:hover{border-color:var(--cyan);transform:translateY(-2px);}
.mc::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--cyan),var(--violet));}
.mv{font-family:'IBM Plex Mono',monospace;font-size:1.9rem;font-weight:700;color:var(--cyan);line-height:1;}
.ml{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-top:5px;}
.ms{font-size:.68rem;color:var(--dim);margin-top:3px;}

/* Section header */
.sh{font-family:'IBM Plex Mono',monospace;font-size:1rem;font-weight:600;color:var(--cyan);
    letter-spacing:.04em;padding:12px 0 7px;border-bottom:1px solid var(--border);margin-bottom:14px;}

/* Kartu AI */
.ac{background:var(--card2);border:1px solid var(--border);border-radius:10px;padding:15px 18px;margin-bottom:10px;transition:border-color .2s;}
.ac:hover{border-color:var(--cyan);}
.at{font-weight:600;font-size:.92rem;color:var(--txt);margin-bottom:5px;}
.ab{font-size:.83rem;color:#94a3b8;line-height:1.5;}

/* Badge */
.badge{display:inline-block;font-size:.66rem;font-weight:600;padding:2px 8px;border-radius:999px;
       text-transform:uppercase;letter-spacing:.05em;margin-right:5px;margin-bottom:3px;}
.bh{background:#3f0f0f;color:var(--coral);border:1px solid var(--coral);}
.bm{background:#3b2c00;color:var(--amber);border:1px solid var(--amber);}
.bl{background:#0d2e1e;color:var(--green);border:1px solid var(--green);}
.bt{background:#1a0b3b;color:#a78bfa;border:1px solid var(--violet);}
.bc{background:#0d1e35;color:var(--cyan);border:1px solid var(--cyan2);}

/* Banner */
.bi{background:#0a1e35;border:1px solid #1a4a7a;border-radius:8px;padding:11px 15px;color:#7dd3fc;font-size:.83rem;}
.bw{background:#1e1500;border:1px solid #6b4c00;border-radius:8px;padding:11px 15px;color:#fcd34d;font-size:.83rem;}
.bs{background:#071f14;border:1px solid #1a5e3a;border-radius:8px;padding:11px 15px;color:#6ee7b7;font-size:.83rem;}
.be{background:#1f0808;border:1px solid #6b1a1a;border-radius:8px;padding:11px 15px;color:#fca5a5;font-size:.83rem;}

/* Upload zone */
.uz{background:var(--card);border:2px dashed var(--border);border-radius:16px;padding:46px;text-align:center;}
.ut{font-family:'IBM Plex Mono',monospace;font-size:1.35rem;color:var(--cyan);margin-bottom:7px;}
.us{color:var(--muted);font-size:.87rem;}

/* Tab override */
.stTabs [data-baseweb="tab-list"]{gap:5px;background:var(--card);border-radius:10px;padding:5px;border:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{border-radius:7px;padding:7px 14px;font-size:.78rem;font-weight:500;color:var(--muted)!important;background:transparent!important;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#0d2640,#1a0b3b)!important;color:var(--cyan)!important;border:1px solid var(--border)!important;}
.stDataFrame,[data-testid="stDataFrame"]{border:1px solid var(--border)!important;border-radius:8px;}
.stButton>button{background:linear-gradient(135deg,#0d2640,#1a0b3b)!important;color:var(--cyan)!important;
  border:1px solid var(--border)!important;border-radius:8px!important;font-family:'IBM Plex Mono',monospace!important;
  font-size:.76rem!important;font-weight:600!important;letter-spacing:.05em!important;padding:7px 16px!important;}
.stButton>button:hover{border-color:var(--cyan)!important;box-shadow:0 0 12px rgba(0,212,255,.18)!important;}
.stSelectbox>div,.stMultiselect>div,.stSlider>div,.stNumberInput>div{background:var(--card2)!important;border-color:var(--border)!important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
_DEFAULTS = dict(
    df=None, df_proc=None, col_types=None, basic_info=None,
    desc_stats=None, missing_df=None, outlier_df=None,
    corr_pearson=None, corr_spearman=None, strong_corrs=None,
    cat_analysis=None, summary=None, ai_result=None,
    dt_result=None, file_name=None, ai_key="",
)
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────
# INISIALISASI MODUL
# ─────────────────────────────────────────────────────────────
loader   = DataLoader()
analyzer = EDAAnalyzer()
viz      = Visualizer()
exporter = ExportUtils()
dt_an    = DecisionTheoryAnalyzer()
dt_viz   = DecisionVisualizer()

def _get_ai(): return AIInsights(api_key=st.session_state.ai_key or None)

# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def mc(val, label, sub="", color="var(--cyan)"):
    return (f'<div class="mc"><div class="mv" style="color:{color}">{val}</div>'
            f'<div class="ml">{label}</div>'
            f'{"<div class=ms>"+sub+"</div>" if sub else ""}</div>')

def badge(txt, k="bc"): return f'<span class="badge {k}">{txt}</span>'

def miss_color(p):
    return "var(--coral)" if p>50 else "#f97316" if p>20 else "var(--amber)" if p>5 else "var(--green)"

def qual_color(s):
    return "var(--green)" if s>=8 else "var(--amber)" if s>=5 else "var(--coral)"

def banner(msg, t="i"):
    cls = {"i":"bi","w":"bw","s":"bs","e":"be"}.get(t,"bi")
    return f'<div class="{cls}">{msg}</div>'

# ─────────────────────────────────────────────────────────────
# PIPELINE ANALISIS
# ─────────────────────────────────────────────────────────────
def run_analysis(df: pd.DataFrame):
    with st.spinner("🔬 Menjalankan pipeline analisis …"):
        col_types, df_proc = loader.detect_column_types(df)
        basic_info         = loader.get_basic_info(df_proc)
        desc_stats         = analyzer.descriptive_stats(df_proc, col_types["numeric"])
        missing_df         = analyzer.missing_value_analysis(df_proc)
        outlier_df         = analyzer.outlier_detection(df_proc, col_types["numeric"])
        corr_p, corr_s     = analyzer.correlation_analysis(df_proc, col_types["numeric"])
        strong_corrs       = analyzer.get_strong_correlations(corr_p)
        cat_analysis       = analyzer.categorical_analysis(df_proc, col_types["categorical"])
        summary            = analyzer.dataset_summary(df_proc, col_types)

        # Decision Theory
        num_cols = col_types.get("numeric", [])
        cat_cols = col_types.get("categorical", [])
        ev_profile   = dt_an.expected_value_profile(df_proc, num_cols)
        risk_profile = dt_an.risk_tolerance_profile(df_proc, num_cols)
        scen_df      = dt_an.scenario_analysis(df_proc, num_cols)
        _, dom_matrix= dt_an.dominance_analysis(df_proc, num_cols)
        dom_pairs, _ = dt_an.dominance_analysis(df_proc, num_cols)
        mcda_df      = dt_an.mcda_weighted_scoring(df_proc, num_cols)
        # Buat matriks keputusan jika ada kolom kategorik & numerik
        dec_matrix = pd.DataFrame()
        if cat_cols and num_cols:
            dec_matrix = dt_an.build_decision_matrix(df_proc, cat_cols[0], num_cols[:6])
        # Sensitivitas: kolom numerik pertama sebagai target
        sens_df = pd.DataFrame()
        if len(num_cols) >= 2:
            sens_df = dt_an.sensitivity_analysis(df_proc, num_cols[0], num_cols[1:])
        dt_result = dict(
            ev_profile=ev_profile, risk_profile=risk_profile,
            scen_df=scen_df, dom_matrix=dom_matrix, dom_pairs=dom_pairs,
            mcda_df=mcda_df, dec_matrix=dec_matrix, sens_df=sens_df,
        )

    st.session_state.update(
        df=df, df_proc=df_proc, col_types=col_types, basic_info=basic_info,
        desc_stats=desc_stats, missing_df=missing_df, outlier_df=outlier_df,
        corr_pearson=corr_p, corr_spearman=corr_s, strong_corrs=strong_corrs,
        cat_analysis=cat_analysis, summary=summary, ai_result=None,
        dt_result=dt_result,
    )

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 8px">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:1.05rem;font-weight:700;color:#00d4ff;letter-spacing:.06em;">🔬 AUTO EDA</div>
      <div style="font-size:.7rem;color:#475569;margin-top:2px;">Dashboard Sains Data Senior</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown("#### 📂 Upload Dataset")
    uploaded = st.file_uploader("CSV atau XLSX", type=["csv","xlsx","xls"],
                                 help="Ukuran maks. rekomendasi: 200 MB",
                                 label_visibility="collapsed")
    if uploaded is not None:
        if uploaded.name != st.session_state.file_name:
            try:
                df_raw = loader.load_file(uploaded)
                st.session_state.file_name = uploaded.name
                run_analysis(df_raw)
                st.success(f"✅ Dimuat: **{uploaded.name}**")
            except Exception as exc:
                st.error(f"❌ Gagal memuat: {exc}")

    st.divider()
    st.markdown("#### 🤖 AI Insight")
    api_key_input = st.text_input("Kunci API Anthropic", type="password",
                                   value=st.session_state.ai_key,
                                   placeholder="sk-ant-…",
                                   help="Diperlukan untuk modul AI Insight")
    if api_key_input != st.session_state.ai_key:
        st.session_state.ai_key = api_key_input
        st.session_state.ai_result = None

    if st.session_state.df_proc is not None and st.session_state.ai_key:
        if st.button("✨ Hasilkan AI Insight", use_container_width=True):
            with st.spinner("🤖 Menghasilkan insight …"):
                ai_engine = _get_ai()
                metadata  = ai_engine.build_metadata(
                    st.session_state.df_proc, st.session_state.col_types,
                    st.session_state.desc_stats, st.session_state.missing_df,
                    st.session_state.strong_corrs,
                )
                st.session_state.ai_result = ai_engine.get_insights(metadata)
            if st.session_state.ai_result.get("success"):
                st.success("✅ Insight siap!")
            else:
                st.error(f"❌ {st.session_state.ai_result.get('error')}")

    if st.session_state.col_types:
        st.divider()
        st.markdown("#### 🎚️ Filter Kolom")
        ct = st.session_state.col_types
        _NF = st.multiselect("Numerik",    ct.get("numeric",[]),    default=ct.get("numeric",[]),    key="fn")
        _CF = st.multiselect("Kategorik",  ct.get("categorical",[]),default=ct.get("categorical",[]),key="fc")
        _DF = st.multiselect("Datetime",   ct.get("datetime",[]),   default=ct.get("datetime",[]),   key="fd")
    else:
        _NF = _CF = _DF = []

    if st.session_state.basic_info:
        bi = st.session_state.basic_info
        st.divider()
        st.markdown(f"""<div style="font-size:.73rem;color:#475569;line-height:2">
          <b style="color:#64748b">File:</b> {st.session_state.file_name or '—'}<br>
          <b style="color:#64748b">Baris:</b> {bi['n_rows']:,}<br>
          <b style="color:#64748b">Kolom:</b> {bi['n_cols']}<br>
          <b style="color:#64748b">Memori:</b> {bi['memory_mb']} MB<br>
          <b style="color:#64748b">Missing:</b> {bi['missing_pct']}%
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HEADER UTAMA
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:24px 0 16px">
  <div style="display:flex;align-items:baseline;gap:12px">
    <span style="font-family:'IBM Plex Mono',monospace;font-size:1.85rem;font-weight:700;color:#00d4ff;letter-spacing:-.01em;">
      AUTO EDA DASHBOARD
    </span>
    <span style="font-size:.76rem;color:#475569;font-family:'IBM Plex Mono',monospace;">
      v3.0 · Bahasa Indonesia
    </span>
  </div>
  <div style="color:#475569;font-size:.86rem;margin-top:5px;">
    Upload dataset CSV atau XLSX manapun — tanpa asumsi nama kolom atau domain data.
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# WELCOME SCREEN (belum upload)
# ─────────────────────────────────────────────────────────────
if st.session_state.df_proc is None:
    st.markdown("""
<div class="uz">
  <div class="ut">← Upload dataset untuk memulai</div>
  <div class="us">
    Mendukung CSV (semua delimiter) dan XLSX · Deteksi encoding otomatis<br>
    Tanpa asumsi nama kolom · Kompatibel dengan domain & skema data apapun<br><br>
    <span style="color:#1e3a5f">● Analisis Statistik</span> &nbsp;
    <span style="color:#1e3a5f">● AI Insight</span> &nbsp;
    <span style="color:#1e3a5f">● Teori Keputusan</span> &nbsp;
    <span style="color:#1e3a5f">● Ekspor Excel</span>
  </div>
</div>""", unsafe_allow_html=True)
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    features = [
        ("📊","Analisis Statistik","Statistik deskriptif, distribusi, outlier, korelasi — otomatis untuk setiap kolom."),
        ("🧠","Teori Keputusan","EV, Maximax, Maximin, Regret, Hurwicz, MCDA, Pareto, Sensitivitas — lengkap."),
        ("🤖","AI Insight","AI mengidentifikasi domain, kualitas data, pola kunci & rekomendasi visualisasi."),
        ("📥","Ekspor Laporan","Unduh laporan Excel multi-sheet lengkap termasuk semua analisis dan AI insight."),
    ]
    for col_w, (ico, ttl, desc) in zip([c1,c2,c3,c4], features):
        col_w.markdown(f"""
<div class="mc" style="text-align:left;padding:18px">
  <div style="font-size:1.5rem;margin-bottom:7px">{ico}</div>
  <div style="font-weight:600;color:#e2e8f0;margin-bottom:4px">{ttl}</div>
  <div style="font-size:.78rem;color:#475569">{desc}</div>
</div>""", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────
# ALIAS SESSION STATE
# ─────────────────────────────────────────────────────────────
df         = st.session_state.df_proc
col_types  = st.session_state.col_types
bi         = st.session_state.basic_info
summary    = st.session_state.summary
desc_stats = st.session_state.desc_stats
missing_df = st.session_state.missing_df
outlier_df = st.session_state.outlier_df
corr_p     = st.session_state.corr_pearson
corr_s     = st.session_state.corr_spearman
strong_c   = st.session_state.strong_corrs
cat_an     = st.session_state.cat_analysis
ai_result  = st.session_state.ai_result
dt         = st.session_state.dt_result or {}

num_cols = _NF if _NF else col_types.get("numeric", [])
cat_cols = _CF if _CF else col_types.get("categorical", [])
dt_cols  = _DF if _DF else col_types.get("datetime", [])

# ─────────────────────────────────────────────────────────────
# TAB UTAMA
# ─────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🏠 Ringkasan",
    "🗂️ Data",
    "📈 Statistik",
    "❓ Missing",
    "📦 Outlier",
    "🔗 Korelasi",
    "🎨 Visualisasi",
    "🧠 Teori Keputusan",
    "🤖 AI Insight",
    "📥 Ekspor",
])

# ══════════════════════════════════════════════════════════════
# TAB 0 — RINGKASAN
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="sh">RINGKASAN DATASET</div>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    kpis = [
        (f"{summary['total_rows']:,}", "Total Baris",        f"{bi['memory_mb']} MB",               "var(--cyan)"),
        (f"{summary['total_cols']}",   "Total Kolom",         f"{summary['numeric_cols']} numerik",  "#7c3aed"),
        (f"{summary['missing_pct']}%", "Data Kosong",         f"{bi['total_missing']:,} sel",        miss_color(summary['missing_pct'])),
        (f"{summary['duplicate_pct']}%","Duplikat",           f"{bi['n_duplicates']:,} baris",       "var(--amber)"),
        (f"{summary['complete_rows_pct']}%","Baris Lengkap",  f"{summary['complete_rows']:,}",       "var(--green)"),
        (f"{summary['total_cols']}",   "Tipe Kolom",          f"{len([k for k,v in col_types.items() if v])} tipe","#ec4899"),
    ]
    for cw,(val,lbl,sub,clr) in zip([c1,c2,c3,c4,c5,c6],kpis):
        cw.markdown(mc(val,lbl,sub,clr), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1,2])
    with left:
        st.plotly_chart(viz.column_type_donut(col_types), use_container_width=True)
    with right:
        st.markdown('<div class="sh">KLASIFIKASI KOLOM</div>', unsafe_allow_html=True)
        icons = {"numeric":"🔢","categorical":"🏷️","datetime":"📅","boolean":"✅",
                 "high_cardinality":"🆔","text":"📝","other":"❓"}
        rows_ = []
        for dtype, cols_ in col_types.items():
            for c_ in cols_:
                rows_.append({
                    "Kolom": c_, "Tipe": dtype, "Ikon": icons.get(dtype,"❓"),
                    "Unik": int(df[c_].nunique()) if c_ in df.columns else "—",
                    "Missing %": f"{df[c_].isna().mean()*100:.1f}%" if c_ in df.columns else "—",
                })
        if rows_:
            st.dataframe(pd.DataFrame(rows_), use_container_width=True, height=340, hide_index=True)

    st.markdown('<div class="sh">DAFTAR KOLOM PER TIPE</div>', unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        for dtype in ("numeric","datetime","boolean"):
            cols_ = col_types.get(dtype,[])
            if cols_:
                pills = " ".join(badge(c_) for c_ in cols_)
                st.markdown(f'<div style="margin-bottom:9px"><span style="color:#64748b;font-size:.72rem;'
                            f'text-transform:uppercase;letter-spacing:.07em">{icons.get(dtype,"")} {dtype}</span>'
                            f'<br>{pills}</div>', unsafe_allow_html=True)
    with cb:
        for dtype in ("categorical","high_cardinality","text"):
            cols_ = col_types.get(dtype,[])
            if cols_:
                pills = " ".join(badge(c_) for c_ in cols_[:20])
                extra = f"… +{len(cols_)-20} lainnya" if len(cols_)>20 else ""
                st.markdown(f'<div style="margin-bottom:9px"><span style="color:#64748b;font-size:.72rem;'
                            f'text-transform:uppercase;letter-spacing:.07em">{icons.get(dtype,"")} {dtype}</span>'
                            f'<br>{pills}<span style="color:#334155;font-size:.73rem">{extra}</span></div>',
                            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 1 — DATA
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="sh">PENJELAJAH DATA MENTAH</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    n_rows_show = c1.number_input("Baris ditampilkan", 5, min(5000,len(df)), 100, step=50)
    col_search  = c2.text_input("🔍 Cari nama kolom","")
    show_dt     = c3.checkbox("Tampilkan baris tipe data", value=False)

    disp_cols = ([c for c in df.columns if col_search.lower() in c.lower()]
                 if col_search else df.columns.tolist())
    disp_df   = df[disp_cols].head(int(n_rows_show))
    if show_dt:
        dtype_row = pd.DataFrame([df[disp_cols].dtypes.astype(str).tolist()], columns=disp_cols)
        disp_df   = pd.concat([dtype_row, disp_df], ignore_index=True)

    st.dataframe(disp_df, use_container_width=True, height=500)
    st.caption(f"Menampilkan {min(len(df),int(n_rows_show)):,} dari {len(df):,} baris · {len(disp_cols)} kolom")

# ══════════════════════════════════════════════════════════════
# TAB 2 — STATISTIK
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="sh">STATISTIK DESKRIPTIF</div>', unsafe_allow_html=True)
    if not desc_stats.empty:
        st.dataframe(
            desc_stats.style.background_gradient(
                subset=[c for c in ["mean","std","skewness","kurtosis"] if c in desc_stats.columns],
                cmap="Blues"),
            use_container_width=True, height=420)

        st.markdown('<div class="sh">ANALISIS MENDALAM PER KOLOM</div>', unsafe_allow_html=True)
        if num_cols:
            sel = st.selectbox("Pilih kolom numerik", num_cols, key="dd")
            st.plotly_chart(viz.single_column_deep_dive(df, sel), use_container_width=True)
            if sel in desc_stats.index:
                row_ = desc_stats.loc[sel]
                html_ = "".join(
                    f'<span style="margin:3px;display:inline-block;background:#0a1628;border:1px solid #1a2f50;'
                    f'border-radius:6px;padding:3px 10px;font-family:IBM Plex Mono,monospace;font-size:.72rem;">'
                    f'<span style="color:#475569">{k}</span> <span style="color:#00d4ff">'
                    f'{round(v,4) if isinstance(v,float) else v}</span></span>'
                    for k,v in row_.items() if pd.notna(v))
                st.markdown(html_, unsafe_allow_html=True)
    else:
        st.markdown(banner("Tidak ada kolom numerik dalam dataset ini.","i"), unsafe_allow_html=True)

    if cat_cols:
        st.markdown('<div class="sh">FREKUENSI KATEGORIK</div>', unsafe_allow_html=True)
        sc = st.selectbox("Pilih kolom kategorik", cat_cols, key="cf")
        if sc in cat_an:
            cc1,cc2 = st.columns([1,2])
            cc1.dataframe(cat_an[sc], use_container_width=True, hide_index=True)
            cc2.plotly_chart(viz.single_cat_chart(df, sc), use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — MISSING
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="sh">ANALISIS NILAI KOSONG (MISSING VALUE)</div>', unsafe_allow_html=True)
    if not missing_df.empty:
        m1,m2,m3,m4 = st.columns(4)
        m1.markdown(mc(f"{bi['total_missing']:,}","Total Sel Kosong","","var(--coral)"), unsafe_allow_html=True)
        m2.markdown(mc(f"{bi['missing_pct']}%","% Missing Keseluruhan","",miss_color(bi['missing_pct'])), unsafe_allow_html=True)
        m3.markdown(mc(str(len(missing_df)),"Kolom dengan Missing","","var(--amber)"), unsafe_allow_html=True)
        m4.markdown(mc(f"{len(df.columns)-len(missing_df)}","Kolom Lengkap","","var(--green)"), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(viz.missing_bar(missing_df), use_container_width=True)
        st.markdown('<div class="sh">POLA MISSING VALUE (HEATMAP)</div>', unsafe_allow_html=True)
        mhr = st.slider("Sampel baris untuk heatmap", 50, min(2000,len(df)), min(500,len(df)), 50)
        st.plotly_chart(viz.missing_pattern_heatmap(df, mhr), use_container_width=True)
        st.markdown('<div class="sh">TABEL DETAIL MISSING VALUE</div>', unsafe_allow_html=True)
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
    else:
        st.markdown(banner("🎉 Dataset ini <b>tidak memiliki nilai kosong</b>!","s"), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 4 — OUTLIER
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="sh">DETEKSI OUTLIER</div>', unsafe_allow_html=True)
    if not outlier_df.empty:
        st.markdown(banner("Metode IQR (aturan 1.5×) dan Z-score (|z| > 3σ) diterapkan pada setiap kolom numerik.","i"), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(viz.outlier_comparison(outlier_df),use_container_width=True,key="outlier_comparison_tab4")
        st.plotly_chart(viz.boxplot_grid(df, num_cols),use_container_width=True,key="outlier_boxplot_tab4")
        st.markdown('<div class="sh">TABEL DETAIL OUTLIER</div>', unsafe_allow_html=True)
        st.dataframe(outlier_df, use_container_width=True, hide_index=True)
    else:
        st.markdown(banner("✅ Tidak ada outlier yang terdeteksi pada kolom numerik.","s"), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 5 — KORELASI
# ══════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="sh">ANALISIS KORELASI</div>', unsafe_allow_html=True)
    if len(num_cols) >= 2:
        met_ = st.radio("Metode korelasi", ["Pearson","Spearman"], horizontal=True, key="cm")
        corr_mat = corr_p if met_=="Pearson" else corr_s
        st.plotly_chart(viz.correlation_heatmap(df, num_cols, met_.lower()), use_container_width=True)
        thr_ = st.slider("Ambang korelasi kuat |r| ≥", 0.0, 1.0, 0.5, 0.05, key="ct")
        sc_ = analyzer.get_strong_correlations(corr_mat, thr_)
        if not sc_.empty:
            st.markdown('<div class="sh">PASANGAN KORELASI KUAT</div>', unsafe_allow_html=True)
            st.dataframe(sc_, use_container_width=True, hide_index=True)
        else:
            st.markdown(banner(f"Tidak ada pasangan dengan |r| ≥ {thr_}","i"), unsafe_allow_html=True)
        if len(num_cols) <= 10:
            st.markdown('<div class="sh">SCATTER MATRIX</div>', unsafe_allow_html=True)
            st.plotly_chart(viz.pairplot(df, num_cols), use_container_width=True)
    else:
        st.markdown(banner("Butuh ≥ 2 kolom numerik untuk analisis korelasi.","i"), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 6 — VISUALISASI
# ══════════════════════════════════════════════════════════════
with tabs[6]:
    vt = st.tabs(["📊 Histogram","📦 Boxplot","🏷️ Kategorik","📅 Time Series"])
    with vt[0]:
        if num_cols:
            mh = st.slider("Maks kolom", 1, min(24,len(num_cols)), min(12,len(num_cols)), key="mh")
            st.plotly_chart(viz.histogram_grid(df, num_cols, mh), use_container_width=True)
        else:
            st.markdown(banner("Tidak ada kolom numerik.","i"), unsafe_allow_html=True)
    with vt[1]:
        if num_cols:
            st.plotly_chart(viz.boxplot_grid(df, num_cols),use_container_width=True,key="visual_boxplot_tab6")
        else:
            st.markdown(banner("Tidak ada kolom numerik.","i"), unsafe_allow_html=True)
    with vt[2]:
        if cat_cols:
            tn = st.slider("Top N nilai per chart", 5, 30, 12, key="tn")
            st.plotly_chart(viz.bar_chart_grid(df, cat_cols, tn), use_container_width=True)
        else:
            st.markdown(banner("Tidak ada kolom kategorik.","i"), unsafe_allow_html=True)
    with vt[3]:
        if dt_cols:
            sd  = st.selectbox("Kolom tanggal/waktu", dt_cols, key="tsd")
            sn  = st.multiselect("Seri numerik", num_cols, default=num_cols[:min(3,len(num_cols))], key="tsn")
            if sn:
                st.plotly_chart(viz.time_series_grid(df, sd, sn), use_container_width=True)
            else:
                st.markdown(banner("Pilih minimal satu kolom numerik.","w"), unsafe_allow_html=True)
        else:
            st.markdown(banner("Tidak ada kolom datetime yang terdeteksi.","i"), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 7 — TEORI PENGAMBILAN KEPUTUSAN
# ══════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown('<div class="sh">🧠 ANALISIS TEORI PENGAMBILAN KEPUTUSAN</div>', unsafe_allow_html=True)

    if not num_cols:
        st.markdown(banner("Butuh minimal 1 kolom numerik untuk analisis keputusan.","w"), unsafe_allow_html=True)
        st.stop()

    ev_profile   = dt.get("ev_profile",   pd.DataFrame())
    risk_profile = dt.get("risk_profile", pd.DataFrame())
    scen_df      = dt.get("scen_df",      pd.DataFrame())
    dom_matrix   = dt.get("dom_matrix",   pd.DataFrame())
    dom_pairs    = dt.get("dom_pairs",    pd.DataFrame())
    mcda_df      = dt.get("mcda_df",      pd.DataFrame())
    dec_matrix   = dt.get("dec_matrix",   pd.DataFrame())
    sens_df      = dt.get("sens_df",      pd.DataFrame())

    # ── Ringkasan Keputusan ──────────────────────────────────
    findings = dt_an.decision_summary(ev_profile, risk_profile, sens_df if not sens_df.empty else None)
    for f in findings:
        tmap = {"success":"s","warning":"w","info":"i","error":"e"}
        st.markdown(
            f'<div class="ac"><div class="at">{f["ikon"]} {f["judul"]}</div>'
            f'<div class="ab">{f["detail"]}</div></div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sub-tab ──────────────────────────────────────────────
    dt_tabs = st.tabs([
        "📊 Nilai Ekspektasi & Risiko",
        "🎯 Matriks Keputusan & Kriteria",
        "📉 Analisis Sensitivitas",
        "⚖️ Dominansi & Pareto",
        "🔮 Skenario & Utilitas",
        "🏅 MCDA Weighted Scoring",
        "📈 CDF & Hurwicz",
    ])

    # ── Sub-tab 0: EV & Risiko ─────────────────────────────
    with dt_tabs[0]:
        st.markdown('<div class="sh">PROFIL NILAI EKSPEKTASI & RISIKO PER VARIABEL</div>', unsafe_allow_html=True)
        st.markdown(banner(
            "Nilai Ekspektasi (EV) = nilai rata-rata yang diharapkan. "
            "Koefisien Variasi (CV%) = ukuran risiko relatif. "
            "Semakin tinggi CV, semakin tidak pasti hasilnya.", "i"),
            unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if not ev_profile.empty:
            # KPI ringkasan risiko
            risk_counts = {"🟢 Rendah":0,"🟡 Sedang":0,"🟠 Tinggi":0,"🔴 Sangat Tinggi":0}
            if "Kategori Risiko" in ev_profile.columns:
                for r in ev_profile["Kategori Risiko"]:
                    if r in risk_counts: risk_counts[r] += 1
            rc1,rc2,rc3,rc4 = st.columns(4)
            clrs = ["var(--green)","var(--amber)","#f97316","var(--coral)"]
            for cw,(lbl,cnt),clr in zip([rc1,rc2,rc3,rc4],risk_counts.items(),clrs):
                cw.markdown(mc(str(cnt), lbl.split(" ",1)[1] if " " in lbl else lbl, "variabel", clr), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            l_, r_ = st.columns([1,1])
            with l_:
                st.markdown('<div class="sh">TABEL PROFIL NILAI EKSPEKTASI</div>', unsafe_allow_html=True)
                st.dataframe(ev_profile, use_container_width=True, hide_index=True, height=340)
            with r_:
                st.plotly_chart(dt_viz.ev_risk_scatter(ev_profile), use_container_width=True)

            # Penjelasan teori
            st.markdown('<div class="sh">📖 KONSEP TEORI</div>', unsafe_allow_html=True)
            st.markdown("""
<div class="ac">
  <div class="at">Nilai Ekspektasi (Expected Value)</div>
  <div class="ab">
    EV = Σ [P(x) × x] — Rata-rata tertimbang semua kemungkinan hasil dengan probabilitasnya.
    Dalam pengambilan keputusan, alternatif dengan EV tertinggi dipilih jika pengambil keputusan bersifat <i>risk neutral</i>.<br><br>
    <b>Koefisien Variasi (CV%)</b> = (σ/μ) × 100 mengukur risiko relatif:
    CV ≤ 15% → Rendah | 15–30% → Sedang | 30–60% → Tinggi | >60% → Sangat Tinggi
  </div>
</div>""", unsafe_allow_html=True)

    # ── Sub-tab 1: Matriks Keputusan & Kriteria ───────────
    with dt_tabs[1]:
        st.markdown('<div class="sh">MATRIKS KEPUTUSAN & KRITERIA KLASIK</div>', unsafe_allow_html=True)
        st.markdown(banner(
            "Matriks keputusan memetakan kondisi alam (baris) × alternatif (kolom). "
            "Pilih kolom skenario dan kolom hasil untuk membangun matriks.", "i"),
            unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        all_cat = col_types.get("categorical",[]) + col_types.get("boolean",[])
        state_col_sel = col_a.selectbox(
            "Kolom Skenario / Kondisi Alam",
            options=all_cat if all_cat else ["— Tidak ada kolom kategorik —"],
            key="dc_state")
        outcome_sel = col_b.multiselect(
            "Kolom Hasil / Payoff (numerik)",
            options=num_cols, default=num_cols[:min(4,len(num_cols))], key="dc_out")
        agg_sel = st.radio("Fungsi agregasi", ["mean","median","sum"], horizontal=True, key="dc_agg")

        if state_col_sel in df.columns and outcome_sel:
            dm = dt_an.build_decision_matrix(df, state_col_sel, outcome_sel, agg_sel)
            if not dm.empty:
                st.markdown('<div class="sh">MATRIKS KEPUTUSAN</div>', unsafe_allow_html=True)
                l_, r_ = st.columns([1,1])
                l_.dataframe(dm, use_container_width=True)
                r_.plotly_chart(dt_viz.decision_matrix_heatmap(dm, f"Agregasi: {agg_sel}"), use_container_width=True)

                # Regret Matrix
                st.markdown('<div class="sh">MATRIKS PENYESALAN (OPPORTUNITY LOSS)</div>', unsafe_allow_html=True)
                maximize_sel = st.checkbox("Maksimasi nilai (centang = benefit, kosong = cost)", value=True, key="dc_max")
                regret = dt_an.get_regret_matrix(dm, maximize_sel)
                rl_, rr_ = st.columns([1,1])
                rl_.dataframe(regret, use_container_width=True)
                rr_.plotly_chart(dt_viz.regret_matrix_heatmap(regret), use_container_width=True)

                # Kriteria Keputusan
                st.markdown('<div class="sh">KRITERIA PENGAMBILAN KEPUTUSAN KLASIK</div>', unsafe_allow_html=True)
                alpha_h = st.slider("Koefisien Optimisme Hurwicz (α)", 0.0, 1.0, 0.5, 0.05, key="hurwicz_a")
                crit_df = dt_an.apply_decision_criteria(dm, hurwicz_alpha=alpha_h, maximize=maximize_sel)

                if not crit_df.empty:
                    st.dataframe(crit_df, use_container_width=True, hide_index=True)
                    st.plotly_chart(dt_viz.criteria_comparison_bar(crit_df), use_container_width=True)

                # Penjelasan kriteria
                st.markdown('<div class="sh">📖 PENJELASAN KRITERIA</div>', unsafe_allow_html=True)
                for krit, desc in [
                    ("🎯 Maximax (Wald Optimis)","Pilih alternatif dengan nilai terbaik tertinggi. Cocok untuk pengambil keputusan <b>sangat optimis</b>."),
                    ("🛡️ Maximin (Wald Pesimis)","Pilih alternatif dengan nilai terburuk terbaik. Strategi <b>konservatif / minimax risk</b>."),
                    ("😔 Minimax Regret (Savage)","Minimalkan penyesalan maksimum. Cocok jika pengambil keputusan takut menyesal memilih yang salah."),
                    ("⚖️ Laplace (Bayes Equal Prob)","Rata-rata semua kondisi alam dengan probabilitas sama. Digunakan saat info probabilitas tidak tersedia."),
                    ("🔀 Hurwicz","α × nilai_terbaik + (1-α) × nilai_terburuk. Parameter α mencerminkan tingkat optimisme (0=pesimis, 1=optimis)."),
                    ("📊 Expected Value (EV)","Σ [P(skenario) × payoff]. Kriteria optimal untuk pengambil keputusan <b>risk neutral</b> jangka panjang."),
                ]:
                    st.markdown(f'<div class="ac"><div class="at">{krit}</div><div class="ab">{desc}</div></div>',
                                unsafe_allow_html=True)
        else:
            st.markdown(banner("Pilih kolom skenario kategorik dan minimal satu kolom hasil numerik.","w"), unsafe_allow_html=True)

    # ── Sub-tab 2: Sensitivitas ────────────────────────────
    with dt_tabs[2]:
        st.markdown('<div class="sh">ANALISIS SENSITIVITAS (WHAT-IF ANALYSIS)</div>', unsafe_allow_html=True)
        st.markdown(banner(
            "Mengukur seberapa besar perubahan variabel input mempengaruhi variabel target. "
            "Elastisitas > 1 = sensitif (elastis); < 1 = tidak sensitif (inelastis).", "i"),
            unsafe_allow_html=True)

        if len(num_cols) >= 2:
            sc1, sc2 = st.columns(2)
            target_c  = sc1.selectbox("Variabel Target (Y)", num_cols, key="sa_target")
            pred_opts = [c for c in num_cols if c != target_c]
            pred_sels = sc2.multiselect("Variabel Prediktor (X)", pred_opts, default=pred_opts[:min(5,len(pred_opts))], key="sa_preds")
            perturb   = st.slider("Besar gangguan (%)", 1, 50, 10, key="sa_perturb")

            if pred_sels:
                with st.spinner("Menghitung sensitivitas …"):
                    s_df = dt_an.sensitivity_analysis(df, target_c, pred_sels, float(perturb))
                if not s_df.empty:
                    l_,r_ = st.columns([1,1])
                    l_.dataframe(s_df, use_container_width=True, hide_index=True, height=360)
                    r_.plotly_chart(dt_viz.sensitivity_tornado(s_df), use_container_width=True)

                    # Highlight variabel paling sensitif
                    top_s = s_df.iloc[0]
                    st.markdown(f"""
<div class="ac">
  <div class="at">🔍 Temuan Sensitivitas Utama</div>
  <div class="ab">
    Variabel <b>{top_s['Prediktor']}</b> memiliki elastisitas tertinggi
    (<b>{top_s['Elastisitas']}</b>) terhadap target <b>{target_c}</b>.
    Perubahan ±{perturb}% pada <b>{top_s['Prediktor']}</b> menggerakkan <b>{target_c}</b>
    sebesar <b>{top_s['Dampak %']}%</b> — {top_s['Tingkat Sensitivitas']}.
    {'Korelasi signifikan secara statistik (p < 0.05).' if top_s['Signifikan (p<0.05)']=='✅ Ya' else 'Korelasi TIDAK signifikan (p ≥ 0.05).'}
  </div>
</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(banner("Tidak cukup data untuk analisis sensitivitas.","w"), unsafe_allow_html=True)
            else:
                st.markdown(banner("Pilih minimal satu variabel prediktor.","w"), unsafe_allow_html=True)
        else:
            st.markdown(banner("Butuh minimal 2 kolom numerik untuk analisis sensitivitas.","w"), unsafe_allow_html=True)

        st.markdown('<div class="sh">📖 KONSEP SENSITIVITAS & ELASTISITAS</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="ac">
  <div class="ab">
    <b>Elastisitas</b> = (ΔY/Y) / (ΔX/X) — mengukur respons proporsional.<br><br>
    🔴 <b>Sangat Sensitif</b> (e ≥ 2): Perubahan kecil X berdampak sangat besar pada Y.<br>
    🟠 <b>Sensitif/Elastis</b> (1 ≤ e < 2): Perubahan X > perubahan Y secara proporsional.<br>
    🟡 <b>Sedang</b> (0.5 ≤ e < 1): Respons sedang.<br>
    🟢 <b>Tidak Sensitif/Inelastis</b> (e < 0.5): Y hampir tidak berubah meski X berubah besar.<br><br>
    Dalam pengambilan keputusan, variabel dengan elastisitas tinggi adalah <b>faktor kritis</b>
    yang harus dimonitor dan dikelola dengan sangat hati-hati.
  </div>
</div>""", unsafe_allow_html=True)

    # ── Sub-tab 3: Dominansi & Pareto ────────────────────
    with dt_tabs[3]:
        st.markdown('<div class="sh">ANALISIS DOMINANSI STOKASTIK</div>', unsafe_allow_html=True)
        st.markdown(banner(
            "Alternatif A mendominasi B jika A ≥ B di semua kondisi alam. "
            "Pareto frontier mengidentifikasi titik-titik efisien (tidak ada yang lebih baik di semua dimensi).", "i"),
            unsafe_allow_html=True)

        if len(num_cols) >= 2:
            da_sel = st.multiselect("Pilih kolom untuk analisis dominansi",
                                     num_cols, default=num_cols[:min(6,len(num_cols))], key="da_cols")
            if len(da_sel) >= 2:
                with st.spinner("Menghitung dominansi …"):
                    dp, dm2 = dt_an.dominance_analysis(df, da_sel)

                l_,r_ = st.columns([1,1])
                with l_:
                    st.markdown('<div class="sh">PASANGAN DOMINANSI</div>', unsafe_allow_html=True)
                    if not dp.empty:
                        st.dataframe(dp, use_container_width=True, hide_index=True, height=320)
                    else:
                        st.markdown(banner("Tidak ada pasangan dominansi.","i"), unsafe_allow_html=True)
                with r_:
                    st.plotly_chart(dt_viz.dominance_heatmap(dm2), use_container_width=True)

            # Pareto Frontier
            st.markdown('<div class="sh">ANALISIS PARETO FRONTIER</div>', unsafe_allow_html=True)
            p1, p2, p3 = st.columns(3)
            x_p = p1.selectbox("Sumbu X (Risiko/Biaya)", num_cols, key="par_x")
            y_p = p2.selectbox("Sumbu Y (Manfaat/Return)", [c for c in num_cols if c != x_p] or num_cols, key="par_y")
            lbl_opts = ["(Tanpa label)"] + (col_types.get("categorical",[]) or [])
            l_p = p3.selectbox("Label titik", lbl_opts, key="par_l")
            label_p = None if l_p.startswith("(") else l_p

            with st.spinner("Menghitung Pareto …"):
                par_df = dt_an.pareto_analysis(df, x_p, y_p, label_p)

            if not par_df.empty:
                n_front = (par_df["Status Pareto"]=="✅ Frontier").sum()
                n_dom   = (par_df["Status Pareto"]=="❌ Terdominasi").sum()
                pa1,pa2 = st.columns(2)
                pa1.markdown(mc(str(n_front),"Titik Pareto Frontier","titik efisien","var(--cyan)"), unsafe_allow_html=True)
                pa2.markdown(mc(str(n_dom),"Titik Terdominasi","dapat diabaikan","var(--muted)"), unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.plotly_chart(dt_viz.pareto_frontier_chart(par_df, x_p, y_p, label_p), use_container_width=True)
                st.dataframe(par_df.head(30), use_container_width=True, hide_index=True)
        else:
            st.markdown(banner("Butuh minimal 2 kolom numerik.","w"), unsafe_allow_html=True)

        st.markdown('<div class="sh">📖 KONSEP DOMINANSI & PARETO</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="ac">
  <div class="ab">
    <b>Dominansi Stokastik Orde-1</b>: Alternatif A mendominasi B jika CDF(A) ≤ CDF(B) untuk semua nilai.
    Artinya A selalu menghasilkan hasil lebih baik atau sama dengan B di setiap kondisi.<br><br>
    <b>Pareto Frontier</b>: Kumpulan titik di mana tidak ada titik lain yang lebih baik di <i>semua</i> dimensi sekaligus.
    Titik-titik pada frontier adalah <b>kandidat keputusan terbaik</b> — pilihan dari frontier bergantung pada preferensi trade-off.
  </div>
</div>""", unsafe_allow_html=True)

    # ── Sub-tab 4: Skenario & Utilitas ───────────────────
    with dt_tabs[4]:
        st.markdown('<div class="sh">ANALISIS SKENARIO (PESIMIS / REALISTIS / OPTIMIS)</div>', unsafe_allow_html=True)
        st.markdown(banner(
            "Skenario berdasarkan persentil P10 (pesimis), P50 (realistis), P90 (optimis). "
            "Membantu pengambil keputusan memahami rentang ketidakpastian.", "i"),
            unsafe_allow_html=True)

        scen_sel = st.multiselect("Pilih variabel untuk analisis skenario",
                                   num_cols, default=num_cols[:min(6,len(num_cols))], key="sc_sel")
        if scen_sel:
            with st.spinner("Menghitung skenario …"):
                sc_df = dt_an.scenario_analysis(df, scen_sel)
            if not sc_df.empty:
                st.plotly_chart(dt_viz.scenario_range_chart(sc_df), use_container_width=True)
                st.dataframe(sc_df, use_container_width=True, hide_index=True)
        else:
            st.markdown(banner("Pilih minimal satu variabel.","w"), unsafe_allow_html=True)

        # Profil Toleransi Risiko
        st.markdown('<div class="sh">PROFIL TOLERANSI RISIKO & FUNGSI UTILITAS</div>', unsafe_allow_html=True)
        st.markdown(banner(
            "Certainty Equivalent (CE) = nilai pasti yang membuat pengambil keputusan sama puasnya "
            "dengan mengambil gamble. Risk Premium = EV - CE.", "i"), unsafe_allow_html=True)

        rtp_sel = st.multiselect("Pilih variabel untuk profil utilitas",
                                  num_cols, default=num_cols[:min(5,len(num_cols))], key="rtp_sel")
        if rtp_sel:
            with st.spinner("Menghitung profil utilitas …"):
                rtp_df = dt_an.risk_tolerance_profile(df, rtp_sel)
            if not rtp_df.empty:
                l_,r_ = st.columns([1,1])
                l_.dataframe(rtp_df, use_container_width=True, hide_index=True, height=320)
                r_.plotly_chart(dt_viz.utility_radar(rtp_df), use_container_width=True)

        st.markdown('<div class="sh">📖 TEORI UTILITAS & TOLERANSI RISIKO</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="ac">
  <div class="ab">
    <b>Fungsi Utilitas</b> mencerminkan preferensi pengambil keputusan terhadap risiko:<br><br>
    ⚖️ <b>Risk Averse</b>: Utilitas konkaf — lebih suka kepastian. CE < EV. Contoh: investor konservatif.<br>
    ⚡ <b>Risk Neutral</b>: Utilitas linear — keputusan berdasarkan EV murni. CE = EV.<br>
    🎲 <b>Risk Seeking</b>: Utilitas konveks — menyukai ketidakpastian demi peluang besar. CE > EV.<br><br>
    <b>Certainty Equivalent (CE)</b>: Nilai pasti yang setara dengan gamble. Dihitung via aproksimasi Arrow-Pratt.
    <b>Risk Premium</b>: EV - CE. Semakin positif = semakin menghindari risiko.
  </div>
</div>""", unsafe_allow_html=True)

    # ── Sub-tab 5: MCDA ───────────────────────────────────
    with dt_tabs[5]:
        st.markdown('<div class="sh">ANALISIS MULTI-KRITERIA (MCDA — WEIGHTED SCORING)</div>', unsafe_allow_html=True)
        st.markdown(banner(
            "MCDA menggabungkan beberapa kriteria dengan bobot kepentingan untuk menghasilkan "
            "skor komposit. Skor 0–1, semakin tinggi semakin baik.", "i"),
            unsafe_allow_html=True)

        if num_cols:
            mcda_sel = st.multiselect("Pilih kriteria (kolom numerik)",
                                       num_cols, default=num_cols[:min(6,len(num_cols))], key="mcda_sel")
            if mcda_sel:
                # Input bobot
                st.markdown('<div class="sh">⚖️ BOBOT KEPENTINGAN PER KRITERIA</div>', unsafe_allow_html=True)
                weight_dict = {}
                benefit_list = []
                ncols_w = min(3, len(mcda_sel))
                wrows   = [mcda_sel[i:i+ncols_w] for i in range(0,len(mcda_sel),ncols_w)]
                for row_w in wrows:
                    wcs = st.columns(len(row_w))
                    for wc, col_ in zip(wcs, row_w):
                        w_ = wc.number_input(f"Bobot: {col_}", 0.0, 10.0, 1.0, 0.1, key=f"w_{col_}")
                        is_ben = wc.checkbox(f"Benefit (↑)", value=True, key=f"b_{col_}")
                        weight_dict[col_] = float(w_)
                        if is_ben: benefit_list.append(col_)

                if st.button("🔄 Hitung Skor MCDA", key="calc_mcda"):
                    with st.spinner("Menghitung MCDA …"):
                        m_df = dt_an.mcda_weighted_scoring(df, mcda_sel, weight_dict, benefit_list)
                    if not m_df.empty:
                        top_n_m = st.slider("Tampilkan top N baris", 5, min(100,len(m_df)), 20, key="mcda_topn")
                        l_,r_ = st.columns([1,1])
                        l_.dataframe(m_df.head(top_n_m), use_container_width=True, hide_index=True, height=380)
                        r_.plotly_chart(dt_viz.mcda_score_chart(m_df, top_n_m), use_container_width=True)

            st.markdown('<div class="sh">📖 METODE MCDA</div>', unsafe_allow_html=True)
            st.markdown("""
<div class="ac">
  <div class="ab">
    <b>Weighted Scoring (Simple Additive Weighting / SAW)</b>:<br>
    Skor = Σ (wᵢ × nilai_ternormalisasi_kriteria_i)<br><br>
    Langkah: (1) Normalisasi setiap kriteria ke [0,1] (min-max). (2) Balik kriteria <i>cost</i> (1 - nilai).
    (3) Kalikan dengan bobot kepentingan. (4) Jumlahkan untuk skor akhir.<br><br>
    <b>Kolom Benefit (↑)</b>: Nilai lebih tinggi = lebih baik (contoh: keuntungan, kualitas).<br>
    <b>Kolom Cost (tidak dicentang)</b>: Nilai lebih rendah = lebih baik (contoh: biaya, risiko).
  </div>
</div>""", unsafe_allow_html=True)

    # ── Sub-tab 6: CDF & Hurwicz ─────────────────────────
    with dt_tabs[6]:
        st.markdown('<div class="sh">FUNGSI DISTRIBUSI KUMULATIF (CDF) — DOMINANSI STOKASTIK</div>', unsafe_allow_html=True)
        st.markdown(banner(
            "CDF memungkinkan perbandingan visual dominansi stokastik. "
            "Jika CDF(A) selalu di bawah CDF(B), maka A mendominasi B secara stokastik.", "i"),
            unsafe_allow_html=True)

        cdf_sel = st.multiselect("Pilih kolom untuk CDF",
                                  num_cols, default=num_cols[:min(5,len(num_cols))], key="cdf_sel")
        if cdf_sel:
            st.plotly_chart(dt_viz.cdf_comparison(df, cdf_sel), use_container_width=True)

        if not dec_matrix.empty:
            st.markdown('<div class="sh">ANALISIS HURWICZ — SENSITIVITAS TERHADAP α</div>', unsafe_allow_html=True)
            st.markdown(banner(
                "Grafik ini menunjukkan bagaimana perubahan koefisien optimisme α (0=pesimis, 1=optimis) "
                "mengubah preferensi di antara alternatif-alternatif.", "i"),
                unsafe_allow_html=True)
            h_max = st.checkbox("Maksimasi (benefit)", value=True, key="h_max2")
            st.plotly_chart(dt_viz.hurwicz_alpha_chart(dec_matrix, h_max), use_container_width=True)
            st.markdown("""
<div class="ac">
  <div class="ab">
    Titik perpotongan antar kurva adalah <b>titik indiferensi</b> — nilai α di mana pengambil keputusan
    berpindah preferensi dari satu alternatif ke alternatif lain.
    Ini sangat berguna untuk analisis <i>break-even</i> optimisme.
  </div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 8 — AI INSIGHT
# ══════════════════════════════════════════════════════════════
with tabs[8]:
    st.markdown('<div class="sh">🤖 INSIGHT BERBASIS AI (CLAUDE)</div>', unsafe_allow_html=True)

    if not st.session_state.ai_key:
        st.markdown(banner("⚠️ Masukkan Kunci API Anthropic di sidebar untuk mengaktifkan AI Insight.","w"), unsafe_allow_html=True)
    elif ai_result is None:
        st.markdown(banner('ℹ️ Klik <b>"✨ Hasilkan AI Insight"</b> di sidebar untuk menganalisis dataset ini dengan Claude.','i'), unsafe_allow_html=True)
    elif not ai_result.get("success"):
        st.markdown(banner(f"❌ {ai_result.get('error','Kesalahan tidak diketahui')}","e"), unsafe_allow_html=True)
    else:
        ins = ai_result["insights"]
        dom  = ins.get("domain",{})
        qual = ins.get("dataset_quality",{})

        d1,d2,d3 = st.columns([2,1,1])
        with d1:
            st.markdown(f"""
<div class="ac">
  <div class="at">🌍 Domain yang Terdeteksi</div>
  <div style="font-size:1.5rem;font-weight:700;color:#00d4ff;margin:7px 0">{dom.get('detected_domain','—')}</div>
  <div class="ab">{dom.get('reasoning','')}</div>
  <div style="margin-top:9px">{badge(dom.get('confidence','').upper(), 'bh' if dom.get('confidence')=='high' else 'bm' if dom.get('confidence')=='medium' else 'bl')}</div>
</div>""", unsafe_allow_html=True)
        with d2:
            sc_ = qual.get("score",0)
            st.markdown(f"""
<div class="ac" style="text-align:center">
  <div class="at">📊 Skor Kualitas</div>
  <div style="font-size:2.8rem;font-weight:800;color:{qual_color(sc_)};margin:7px 0">
    {sc_}<span style="font-size:1.1rem;color:#475569">/10</span>
  </div>
  <div style="color:#64748b;font-size:.83rem">{qual.get('label','')}</div>
</div>""", unsafe_allow_html=True)
        with d3:
            st.markdown(f"""
<div class="ac">
  <div class="at">✅ Kelebihan</div>
  {"".join(f'<div class="ab" style="margin-bottom:3px">• {s}</div>' for s in qual.get("strengths",[])[:3])}
  <div class="at" style="margin-top:9px">⚠️ Perhatian</div>
  {"".join(f'<div class="ab" style="margin-bottom:3px">• {c}</div>' for c in qual.get("concerns",[])[:3])}
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="sh">TEMUAN UTAMA</div>', unsafe_allow_html=True)
        for ki in ins.get("key_insights",[]):
            imp   = ki.get("importance","low")
            bmap  = {"high":"bh","medium":"bm","low":"bl"}
            cols_ = " ".join(badge(c,"bc") for c in ki.get("affected_columns",[]))
            st.markdown(f"""
<div class="ac">
  <div class="at">{badge(imp.upper(), bmap.get(imp,"bl"))} {badge(ki.get('type',''),"bt")} {ki.get('title','')}</div>
  <div class="ab">{ki.get('description','')}</div>
  {"<div style='margin-top:7px'>"+cols_+"</div>" if cols_ else ""}
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="sh">REKOMENDASI VISUALISASI</div>', unsafe_allow_html=True)
        rv_list = ins.get("recommended_visualisations",[])
        if rv_list:
            rvc = st.columns(min(3,len(rv_list)))
            for i,rv in enumerate(rv_list):
                p = rv.get("priority","low")
                with rvc[i%len(rvc)]:
                    st.markdown(f"""
<div class="ac">
  <div class="at">{badge(p.upper(), 'bh' if p=='high' else 'bm' if p=='medium' else 'bl')} 📊 {rv.get('chart_type','')}</div>
  <div class="ab">{rv.get('reason','')}</div>
  <div style="margin-top:7px">{"".join(badge(c,"bc") for c in rv.get("columns",[]))}</div>
</div>""", unsafe_allow_html=True)

        dqi = ins.get("data_quality_issues",[])
        if dqi:
            st.markdown('<div class="sh">MASALAH KUALITAS DATA</div>', unsafe_allow_html=True)
            for issue in dqi:
                sv = issue.get("severity","low")
                st.markdown(f"""
<div class="ac">
  <div class="at">{badge(sv.upper(),'bh' if sv=='high' else 'bm' if sv=='medium' else 'bl')} {issue.get('issue','')}</div>
  <div class="ab"><b>Rekomendasi:</b> {issue.get('recommendation','')}<br>
  {"".join(badge(c,"bc") for c in issue.get("affected_columns",[]))}</div>
</div>""", unsafe_allow_html=True)

        bq_c, ns_c = st.columns(2)
        with bq_c:
            bqs = ins.get("business_questions",[])
            if bqs:
                st.markdown('<div class="sh">💡 PERTANYAAN BISNIS</div>', unsafe_allow_html=True)
                for q in bqs:
                    st.markdown(f'<div class="ac"><div class="ab">❓ {q}</div></div>', unsafe_allow_html=True)
        with ns_c:
            nss = ins.get("next_steps",[])
            if nss:
                st.markdown('<div class="sh">🚀 LANGKAH SELANJUTNYA</div>', unsafe_allow_html=True)
                for i,ns_ in enumerate(nss,1):
                    st.markdown(f'<div class="ac"><div class="ab"><span style="color:#00d4ff;font-weight:600">{i}.</span> {ns_}</div></div>',
                                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 9 — EKSPOR
# ══════════════════════════════════════════════════════════════
with tabs[9]:
    st.markdown('<div class="sh">EKSPOR & UNDUH LAPORAN</div>', unsafe_allow_html=True)
    base_name = (st.session_state.file_name or "dataset").rsplit(".",1)[0]

    st.markdown("""
<div class="ac">
  <div class="at">📊 Laporan Excel Lengkap (.xlsx)</div>
  <div class="ab">
    Workbook multi-sheet: Sampel Dataset · Tipe Kolom · Statistik Deskriptif ·
    Nilai Kosong · Deteksi Outlier · Matriks Korelasi ·
    Frekuensi Kategorik · AI Insight (jika tersedia).
  </div>
</div>""", unsafe_allow_html=True)

    if st.button("⬇️ Unduh Laporan Excel Lengkap"):
        with st.spinner("Membangun laporan Excel …"):
            xlsx = exporter.to_excel_report(
                df=df, desc_stats=desc_stats, missing_df=missing_df,
                outlier_df=outlier_df, corr_pearson=corr_p,
                cat_analysis=cat_an, col_types=col_types, ai_insights=ai_result,
            )
        st.download_button(
            label="💾 Simpan Laporan Excel",
            data=xlsx,
            file_name=f"{base_name}_laporan_eda.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.markdown("---")
    st.markdown('<div class="sh">UNDUH TABEL INDIVIDUAL (CSV)</div>', unsafe_allow_html=True)

    tables_ = {
        "📋 Statistik Deskriptif":    desc_stats.reset_index().rename(columns={"index":"Kolom"}) if not desc_stats.empty else pd.DataFrame(),
        "❓ Nilai Kosong":             missing_df,
        "📦 Deteksi Outlier":          outlier_df,
        "🔗 Korelasi (Pearson)":       corr_p,
        "🔗 Korelasi Kuat":            strong_c,
        "📊 Profil Nilai Ekspektasi":  dt.get("ev_profile", pd.DataFrame()),
        "⚖️ Profil Toleransi Risiko":  dt.get("risk_profile", pd.DataFrame()),
        "🔮 Analisis Skenario":        dt.get("scen_df", pd.DataFrame()),
        "🗂️ Dataset Terproses (penuh)": df,
    }
    dl_cols = st.columns(3)
    for i,(lbl, tbl) in enumerate(tables_.items()):
        fname_ = lbl.split(" ",1)[-1].lower().replace(" ","_").replace("(","").replace(")","").replace("/","_")
        with dl_cols[i%3]:
            if isinstance(tbl, pd.DataFrame) and not tbl.empty:
                st.download_button(
                    label=f"⬇️ {lbl}",
                    data=exporter.to_csv(tbl),
                    file_name=f"{base_name}_{fname_}.csv",
                    mime="text/csv", key=f"dl_{fname_}_{i}",
                )
            else:
                st.button(f"⬇️ {lbl} (kosong)", disabled=True, key=f"dis_{fname_}_{i}")

    if ai_result and ai_result.get("success"):
        st.markdown("---")
        st.download_button(
            label="⬇️ 🤖 AI Insight (JSON)",
            data=exporter.insights_to_json(ai_result["insights"]),
            file_name=f"{base_name}_ai_insight.json",
            mime="application/json", key="dl_ai",
        )

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:28px 0 14px;font-size:.7rem;color:#1e293b;
            font-family:'IBM Plex Mono',monospace;letter-spacing:.06em;">
  AUTO EDA DASHBOARD v3.0 · Streamlit + Plotly + AI · Bahasa Indonesia
</div>
""", unsafe_allow_html=True)
