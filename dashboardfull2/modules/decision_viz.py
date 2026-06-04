"""
modules/decision_viz.py
────────────────────────────────────────────────────────────────
Visualisasi Teori Pengambilan Keputusan
Semua chart Plotly untuk modul decision_theory.py
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Tema warna konsisten ──────────────────────────────────────
_BG      = "rgba(0,0,0,0)"
_TMPL    = "plotly_dark"
_CYAN    = "#00d4ff"
_CORAL   = "#ff6b6b"
_EMERALD = "#10b981"
_AMBER   = "#f59e0b"
_VIOLET  = "#7c3aed"
_PALETTE = [
    "#00d4ff","#7c3aed","#10b981","#f59e0b","#ef4444",
    "#ec4899","#14b8a6","#f97316","#6366f1","#84cc16",
]

def _layout(**kw) -> dict:
    base = dict(
        template=_TMPL,
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family="IBM Plex Sans, sans-serif", size=12, color="#94a3b8"),
        margin=dict(t=70, b=60, l=60, r=30),
    )
    base.update(kw)
    return base

def _empty(msg: str, height: int = 280) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(size=13, color="#475569"),
    )
    fig.update_layout(
        **_layout(height=height),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


class DecisionVisualizer:
    """Semua chart visualisasi teori pengambilan keputusan."""

    # ══════════════════════════════════════════════════════════════
    # 1. PROFIL RISIKO — Scatter EV vs Risk
    # ══════════════════════════════════════════════════════════════

    def ev_risk_scatter(self, ev_df: pd.DataFrame) -> go.Figure:
        """
        Scatter plot: Nilai Ekspektasi (x) vs Simpangan Baku (y).
        Kuadran menunjukkan profil keputusan.
        """
        if ev_df.empty:
            return _empty("Tidak ada data profil risiko.")

        required = {"Nilai Ekspektasi (EV)", "Simpangan Baku (σ)", "Kolom"}
        if not required.issubset(ev_df.columns):
            return _empty("Kolom profil risiko tidak lengkap.")

        ev   = pd.to_numeric(ev_df["Nilai Ekspektasi (EV)"], errors="coerce")
        std  = pd.to_numeric(ev_df["Simpangan Baku (σ)"],   errors="coerce")
        cols = ev_df["Kolom"]

        color_map = {
            "🟢 Rendah":        _EMERALD,
            "🟡 Sedang":        _AMBER,
            "🟠 Tinggi":        "#f97316",
            "🔴 Sangat Tinggi": _CORAL,
        }
        colors = [
            color_map.get(ev_df["Kategori Risiko"].iloc[i], _CYAN)
            for i in range(len(ev_df))
        ] if "Kategori Risiko" in ev_df.columns else [_CYAN] * len(ev_df)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ev, y=std,
            mode="markers+text",
            text=cols,
            textposition="top center",
            textfont=dict(size=10, color="#94a3b8"),
            marker=dict(
                size=14, color=colors,
                line=dict(width=1, color="#1a2f50"),
                symbol="circle",
            ),
            hovertemplate="<b>%{text}</b><br>EV: %{x:.4f}<br>σ: %{y:.4f}<extra></extra>",
        ))

        # Garis referensi kuadran
        if ev.notna().any() and std.notna().any():
            ev_mid  = float(ev.mean())
            std_mid = float(std.mean())
            fig.add_hline(y=std_mid,  line_dash="dot", line_color="#334155",
                          annotation_text="Rata-rata Risiko", annotation_font_color="#475569")
            fig.add_vline(x=ev_mid,   line_dash="dot", line_color="#334155",
                          annotation_text="Rata-rata EV", annotation_font_color="#475569")

        fig.update_layout(**_layout(
            height=460,
            title_text="<b>Peta Risiko — Nilai Ekspektasi vs Simpangan Baku</b>",
            title_font_size=14,
            xaxis_title="Nilai Ekspektasi (EV)",
            yaxis_title="Simpangan Baku (Risiko)",
            showlegend=False,
        ))
        return fig

    # ══════════════════════════════════════════════════════════════
    # 2. MATRIKS KEPUTUSAN — Heatmap
    # ══════════════════════════════════════════════════════════════

    def decision_matrix_heatmap(self, matrix: pd.DataFrame, title: str = "") -> go.Figure:
        """Heatmap berwarna dari matriks keputusan."""
        if matrix.empty:
            return _empty("Matriks keputusan kosong.")

        fig = go.Figure(go.Heatmap(
            z=matrix.values.astype(float),
            x=[str(c) for c in matrix.columns],
            y=[str(i) for i in matrix.index],
            colorscale="RdYlGn",
            text=np.round(matrix.values.astype(float), 3),
            texttemplate="%{text}",
            textfont=dict(size=10),
            colorbar=dict(title="Nilai", thickness=14, len=0.8),
            hoverongaps=False,
        ))
        fig.update_layout(**_layout(
            height=max(340, len(matrix) * 44 + 140),
            title_text=f"<b>Matriks Keputusan{' — ' + title if title else ''}</b>",
            title_font_size=14,
            xaxis=dict(tickangle=-35, title="Alternatif"),
            yaxis=dict(title="Kondisi Alam / Skenario"),
        ))
        return fig

    # ══════════════════════════════════════════════════════════════
    # 3. MATRIKS PENYESALAN — Heatmap
    # ══════════════════════════════════════════════════════════════

    def regret_matrix_heatmap(self, regret: pd.DataFrame) -> go.Figure:
        """Heatmap matriks penyesalan (opportunity loss)."""
        if regret.empty:
            return _empty("Matriks penyesalan kosong.")

        fig = go.Figure(go.Heatmap(
            z=regret.values.astype(float),
            x=[str(c) for c in regret.columns],
            y=[str(i) for i in regret.index],
            colorscale="Reds",
            text=np.round(regret.values.astype(float), 3),
            texttemplate="%{text}",
            textfont=dict(size=10),
            colorbar=dict(title="Penyesalan", thickness=14, len=0.8),
        ))
        fig.update_layout(**_layout(
            height=max(340, len(regret) * 44 + 140),
            title_text="<b>Matriks Penyesalan (Opportunity Loss / Regret)</b>",
            title_font_size=14,
            xaxis=dict(tickangle=-35, title="Alternatif"),
            yaxis=dict(title="Kondisi Alam"),
        ))
        return fig

    # ══════════════════════════════════════════════════════════════
    # 4. KRITERIA KEPUTUSAN — Grouped Bar
    # ══════════════════════════════════════════════════════════════

    def criteria_comparison_bar(self, crit_df: pd.DataFrame) -> go.Figure:
        """Bar chart perbandingan nilai per kriteria keputusan."""
        if crit_df.empty or "Alternatif" not in crit_df.columns:
            return _empty("Tidak ada data kriteria keputusan.")

        criteria_cols = [c for c in crit_df.columns if c != "Alternatif"]
        if not criteria_cols:
            return _empty("Tidak ada kolom kriteria.")

        # Bersihkan tanda ★
        plot_df = crit_df.copy()
        for col in criteria_cols:
            plot_df[col] = pd.to_numeric(
                plot_df[col].astype(str).str.replace("★ ", "", regex=False),
                errors="coerce"
            )

        fig = go.Figure()
        for i, crit in enumerate(criteria_cols):
            fig.add_trace(go.Bar(
                name=crit,
                x=plot_df["Alternatif"].astype(str),
                y=plot_df[crit],
                marker_color=_PALETTE[i % len(_PALETTE)],
                text=plot_df[crit].round(3),
                textposition="auto",
            ))

        fig.update_layout(**_layout(
            height=460,
            title_text="<b>Perbandingan Nilai per Kriteria Keputusan</b>",
            title_font_size=14,
            barmode="group",
            xaxis_title="Alternatif",
            yaxis_title="Nilai",
            legend=dict(bgcolor="rgba(0,0,0,0.4)", font_size=10),
        ))
        return fig

    # ══════════════════════════════════════════════════════════════
    # 5. ANALISIS SENSITIVITAS — Tornado Chart
    # ══════════════════════════════════════════════════════════════

    def sensitivity_tornado(self, sens_df: pd.DataFrame) -> go.Figure:
        """Tornado chart dampak sensitivitas per prediktor."""
        if sens_df.empty or "Prediktor" not in sens_df.columns:
            return _empty("Tidak ada data sensitivitas.")

        if "Dampak %" not in sens_df.columns:
            return _empty("Kolom 'Dampak %' tidak ditemukan.")

        plot_df = sens_df.sort_values("Dampak %", key=abs, ascending=True).tail(15)
        values  = pd.to_numeric(plot_df["Dampak %"], errors="coerce").fillna(0)
        colors  = [_CORAL if v >= 0 else _CYAN for v in values]

        fig = go.Figure(go.Bar(
            x=values,
            y=plot_df["Prediktor"].astype(str),
            orientation="h",
            marker_color=colors,
            text=values.round(2).astype(str) + "%",
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Dampak: %{x:.2f}%<extra></extra>",
        ))
        fig.add_vline(x=0, line_color="#475569", line_width=1)
        fig.update_layout(**_layout(
            height=max(360, len(plot_df) * 32 + 140),
            title_text="<b>Tornado Chart — Dampak Sensitivitas pada Target (%)</b>",
            title_font_size=14,
            xaxis_title="Dampak pada Target (%)",
            yaxis_title="Prediktor",
        ))
        return fig

    # ══════════════════════════════════════════════════════════════
    # 6. ANALISIS PARETO — Scatter Frontier
    # ══════════════════════════════════════════════════════════════

    def pareto_frontier_chart(
        self,
        pareto_df: pd.DataFrame,
        x_col: str,
        y_col: str,
        label_col: Optional[str] = None,
    ) -> go.Figure:
        """Scatter chart Pareto frontier."""
        if pareto_df.empty:
            return _empty("Data Pareto tidak tersedia.")
        if x_col not in pareto_df.columns or y_col not in pareto_df.columns:
            return _empty(f"Kolom '{x_col}' atau '{y_col}' tidak ditemukan.")

        frontier  = pareto_df[pareto_df["Status Pareto"] == "✅ Frontier"]
        dominated = pareto_df[pareto_df["Status Pareto"] == "❌ Terdominasi"]

        fig = go.Figure()

        # Titik terdominasi
        if not dominated.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_numeric(dominated[x_col], errors="coerce"),
                y=pd.to_numeric(dominated[y_col], errors="coerce"),
                mode="markers",
                name="Terdominasi",
                marker=dict(size=8, color="#334155", symbol="circle"),
                text=dominated[label_col].astype(str) if label_col and label_col in dominated.columns else None,
                hovertemplate="<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>",
            ))

        # Titik frontier
        if not frontier.empty:
            front_sorted = frontier.sort_values(x_col)
            fig.add_trace(go.Scatter(
                x=pd.to_numeric(front_sorted[x_col], errors="coerce"),
                y=pd.to_numeric(front_sorted[y_col], errors="coerce"),
                mode="markers+lines",
                name="Pareto Frontier",
                marker=dict(size=12, color=_CYAN, symbol="star",
                            line=dict(width=1, color="#0a1628")),
                line=dict(color=_CYAN, width=2, dash="dot"),
                text=front_sorted[label_col].astype(str) if label_col and label_col in front_sorted.columns else None,
                hovertemplate="<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>",
            ))

        fig.update_layout(**_layout(
            height=460,
            title_text=f"<b>Pareto Frontier — {x_col} vs {y_col}</b>",
            title_font_size=14,
            xaxis_title=x_col,
            yaxis_title=y_col,
            legend=dict(bgcolor="rgba(0,0,0,0.4)"),
        ))
        return fig

    # ══════════════════════════════════════════════════════════════
    # 7. PROFIL UTILITAS — Radar / Spider Chart
    # ══════════════════════════════════════════════════════════════

    def utility_radar(self, risk_df: pd.DataFrame) -> go.Figure:
        """Radar chart profil utilitas & toleransi risiko per variabel."""
        if risk_df.empty:
            return _empty("Data profil utilitas tidak tersedia.")

        numeric_fields = ["EV", "CV %", "Risk Premium", "VaR 5%"]
        available = [f for f in numeric_fields if f in risk_df.columns]
        if len(available) < 3:
            return _empty("Butuh minimal 3 metrik numerik untuk radar chart.")

        cols_to_show = risk_df["Kolom"].tolist()[:8]
        fig = go.Figure()

        for i, col_name in enumerate(cols_to_show):
            row = risk_df[risk_df["Kolom"] == col_name].iloc[0]
            vals = []
            for field in available:
                v = pd.to_numeric(row[field], errors="coerce")
                vals.append(float(v) if pd.notna(v) else 0.0)

            # Normalisasi per field
            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=available + [available[0]],
                fill="toself",
                name=col_name,
                line=dict(color=_PALETTE[i % len(_PALETTE)], width=2),
                fillcolor=f"rgba({_hex_rgb(_PALETTE[i % len(_PALETTE)])},0.08)",
            ))

        fig.update_layout(**_layout(
            height=480,
            title_text="<b>Radar Profil Utilitas & Risiko</b>",
            title_font_size=14,
            polar=dict(
                bgcolor="#050d1a",
                radialaxis=dict(visible=True, color="#334155"),
                angularaxis=dict(color="#475569"),
            ),
            legend=dict(bgcolor="rgba(0,0,0,0.4)", font_size=10),
        ))
        return fig

    # ══════════════════════════════════════════════════════════════
    # 8. ANALISIS SKENARIO — Waterfall / Range Chart
    # ══════════════════════════════════════════════════════════════

    def scenario_range_chart(self, scen_df: pd.DataFrame) -> go.Figure:
        """Chart rentang skenario Pesimis–Realistis–Optimis per variabel."""
        if scen_df.empty or "Kolom" not in scen_df.columns:
            return _empty("Data skenario tidak tersedia.")

        # Cari kolom skenario
        pess_col = next((c for c in scen_df.columns if "Pesimis" in c and "Deviasi" not in c), None)
        real_col = next((c for c in scen_df.columns if "Realistis" in c and "Deviasi" not in c), None)
        opti_col = next((c for c in scen_df.columns if "Optimis" in c and "Deviasi" not in c), None)
        ev_col   = "Nilai Ekspektasi"

        if not all([pess_col, real_col, opti_col]):
            return _empty("Kolom skenario (Pesimis/Realistis/Optimis) tidak ditemukan.")

        labels = scen_df["Kolom"].astype(str).tolist()
        pess   = pd.to_numeric(scen_df[pess_col], errors="coerce").tolist()
        real   = pd.to_numeric(scen_df[real_col], errors="coerce").tolist()
        opti   = pd.to_numeric(scen_df[opti_col], errors="coerce").tolist()
        ev     = pd.to_numeric(scen_df[ev_col],   errors="coerce").tolist() if ev_col in scen_df.columns else real

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Pesimis (P10)",
            x=labels, y=pess,
            marker_color=_CORAL, opacity=0.85,
        ))
        fig.add_trace(go.Bar(
            name="Realistis (P50)",
            x=labels, y=real,
            marker_color=_AMBER, opacity=0.85,
        ))
        fig.add_trace(go.Bar(
            name="Optimis (P90)",
            x=labels, y=opti,
            marker_color=_EMERALD, opacity=0.85,
        ))
        fig.add_trace(go.Scatter(
            name="Nilai Ekspektasi",
            x=labels, y=ev,
            mode="markers+lines",
            marker=dict(size=10, color=_CYAN, symbol="diamond"),
            line=dict(color=_CYAN, width=2, dash="dot"),
        ))
        fig.update_layout(**_layout(
            height=460,
            title_text="<b>Analisis Skenario — Pesimis / Realistis / Optimis</b>",
            title_font_size=14,
            barmode="group",
            xaxis_title="Variabel",
            yaxis_title="Nilai",
            legend=dict(bgcolor="rgba(0,0,0,0.4)", font_size=10),
        ))
        return fig

    # ══════════════════════════════════════════════════════════════
    # 9. MCDA — Horizontal Score Bar
    # ══════════════════════════════════════════════════════════════

    def mcda_score_chart(self, mcda_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
        """Bar chart horizontal skor MCDA teratas."""
        if mcda_df.empty or "Skor MCDA" not in mcda_df.columns:
            return _empty("Data skor MCDA tidak tersedia.")

        plot_df = mcda_df.head(top_n).copy()
        # Buat label dari index
        plot_df["Label"] = [f"Baris {i+1}" for i in range(len(plot_df))]

        colors = [
            _EMERALD if s >= 0.7 else _AMBER if s >= 0.4 else _CORAL
            for s in plot_df["Skor MCDA"]
        ]

        fig = go.Figure(go.Bar(
            x=plot_df["Skor MCDA"],
            y=plot_df["Label"],
            orientation="h",
            marker_color=colors,
            text=plot_df["Skor MCDA"].round(4),
            textposition="auto",
        ))
        fig.update_layout(**_layout(
            height=max(360, len(plot_df) * 28 + 140),
            title_text=f"<b>Skor MCDA — Top {top_n} Baris Terbaik</b>",
            title_font_size=14,
            xaxis_title="Skor MCDA (0–1)",
            yaxis=dict(autorange="reversed"),
        ))
        return fig

    # ══════════════════════════════════════════════════════════════
    # 10. DOMINANSI — Heatmap Matriks
    # ══════════════════════════════════════════════════════════════

    def dominance_heatmap(self, dom_matrix: pd.DataFrame) -> go.Figure:
        """Heatmap biner matriks dominansi antar alternatif."""
        if dom_matrix.empty:
            return _empty("Matriks dominansi kosong.")

        labels = [str(c) for c in dom_matrix.columns]
        fig = go.Figure(go.Heatmap(
            z=dom_matrix.values.astype(float),
            x=labels,
            y=labels,
            colorscale=[[0, "#0a1628"], [1, _CYAN]],
            text=dom_matrix.values.astype(int),
            texttemplate="%{text}",
            textfont=dict(size=11, color="#e2e8f0"),
            showscale=False,
            hoverongaps=False,
        ))
        fig.update_layout(**_layout(
            height=max(340, len(labels) * 44 + 140),
            title_text="<b>Matriks Dominansi (1 = A mendominasi B)</b>",
            title_font_size=14,
            xaxis=dict(tickangle=-35, title="Alternatif B"),
            yaxis=dict(title="Alternatif A"),
        ))
        return fig

    # ══════════════════════════════════════════════════════════════
    # 11. HURWICZ — Sensitivitas terhadap α
    # ══════════════════════════════════════════════════════════════

    def hurwicz_alpha_chart(
        self, matrix: pd.DataFrame, maximize: bool = True
    ) -> go.Figure:
        """
        Grafik skor Hurwicz untuk setiap alternatif
        pada rentang α = 0 → 1 (pesimis → optimis).
        """
        if matrix.empty:
            return _empty("Matriks keputusan kosong untuk analisis Hurwicz.")

        alphas = np.linspace(0, 1, 51)
        fig    = go.Figure()

        for i, alt in enumerate(matrix.columns):
            vals    = matrix[alt].values.astype(float)
            best_v  = vals.max() if maximize else vals.min()
            worst_v = vals.min() if maximize else vals.max()
            scores  = alphas * best_v + (1 - alphas) * worst_v

            fig.add_trace(go.Scatter(
                x=alphas,
                y=scores,
                mode="lines",
                name=str(alt),
                line=dict(color=_PALETTE[i % len(_PALETTE)], width=2),
            ))

        fig.add_vline(x=0.5, line_dash="dash", line_color="#475569",
                      annotation_text="α=0.5 (Netral)",
                      annotation_font_color="#64748b")
        fig.update_layout(**_layout(
            height=420,
            title_text="<b>Analisis Hurwicz — Skor per Nilai α (0=Pesimis, 1=Optimis)</b>",
            title_font_size=14,
            xaxis_title="Koefisien Optimisme (α)",
            yaxis_title="Skor Hurwicz",
            legend=dict(bgcolor="rgba(0,0,0,0.4)", font_size=10),
        ))
        return fig

    # ══════════════════════════════════════════════════════════════
    # 12. DISTRIBUSI KUMULATIF — CDF Perbandingan
    # ══════════════════════════════════════════════════════════════

    def cdf_comparison(
        self, df: pd.DataFrame, numeric_cols: List[str], max_cols: int = 6
    ) -> go.Figure:
        """CDF kumulatif semua kolom numerik — berguna untuk stochastic dominance."""
        cols = [c for c in numeric_cols if c in df.columns][:max_cols]
        if not cols:
            return _empty("Tidak ada kolom numerik untuk CDF.")

        fig = go.Figure()
        for i, col in enumerate(cols):
            s      = df[col].dropna().sort_values()
            cdf    = np.arange(1, len(s) + 1) / len(s)
            fig.add_trace(go.Scatter(
                x=s, y=cdf * 100,
                mode="lines",
                name=col,
                line=dict(color=_PALETTE[i % len(_PALETTE)], width=2),
            ))

        fig.update_layout(**_layout(
            height=440,
            title_text="<b>Fungsi Distribusi Kumulatif (CDF) — Perbandingan Alternatif</b>",
            title_font_size=14,
            xaxis_title="Nilai",
            yaxis_title="Probabilitas Kumulatif (%)",
            legend=dict(bgcolor="rgba(0,0,0,0.4)", font_size=10),
        ))
        return fig


def _hex_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"{r},{g},{b}"
