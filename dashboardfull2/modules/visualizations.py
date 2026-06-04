"""
modules/visualizations.py
─────────────────────────────────────────────
All Plotly chart generators for the EDA dashboard.
Every function is robust to edge cases and
never assumes specific column names.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────
# THEME CONSTANTS
# ─────────────────────────────────────────────────────────────

_BG     = "rgba(0,0,0,0)"
_TMPL   = "plotly_dark"
_CYAN   = "#00d4ff"
_CORAL  = "#ff6b6b"
_PALETTE = [
    "#00d4ff", "#7c3aed", "#10b981", "#f59e0b", "#ef4444",
    "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#84cc16",
]

def _base_layout(**kwargs) -> dict:
    return dict(
        template=_TMPL,
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family="IBM Plex Sans, sans-serif", size=12, color="#94a3b8"),
        margin=dict(t=60, b=60, l=60, r=20),
        **kwargs,
    )


class Visualizer:
    """Generates all EDA visualisations as Plotly figures."""

    # ─────────────────────────────────────────────────────────────
    # HISTOGRAMS
    # ─────────────────────────────────────────────────────────────

    def histogram_grid(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        max_plots: int = 12,
    ) -> go.Figure:
        """Grid of histograms for up to max_plots numeric columns."""
        cols = [c for c in numeric_cols if c in df.columns][:max_plots]
        if not cols:
            return self._empty("No numeric columns available.")

        n_cols = min(3, len(cols))
        n_rows = (len(cols) + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=cols,
            vertical_spacing=0.08,
            horizontal_spacing=0.06,
        )

        for idx, col in enumerate(cols):
            r, c = divmod(idx, n_cols)
            s = df[col].dropna()
            if len(s) == 0:
                continue
            fig.add_trace(
                go.Histogram(
                    x=s,
                    nbinsx=min(40, max(10, len(s.unique()))),
                    marker_color=_PALETTE[idx % len(_PALETTE)],
                    opacity=0.80,
                    showlegend=False,
                    name=col,
                ),
                row=r + 1, col=c + 1,
            )

        fig.update_layout(
            **_base_layout(
                height=320 * n_rows,
                title_text="<b>Distributions — Numeric Columns</b>",
                title_font_size=15,
                showlegend=False,
            )
        )
        return fig

    # ─────────────────────────────────────────────────────────────
    # BOXPLOTS
    # ─────────────────────────────────────────────────────────────

    def boxplot_grid(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
    ) -> go.Figure:
        """Side-by-side boxplots for all numeric columns."""
        cols = [c for c in numeric_cols if c in df.columns]
        if not cols:
            return self._empty("No numeric columns available.")

        fig = go.Figure()
        for i, col in enumerate(cols):
            s = df[col].dropna()
            if len(s) == 0:
                continue
            fig.add_trace(
                go.Box(
                    y=s,
                    name=col,
                    marker_color=_PALETTE[i % len(_PALETTE)],
                    boxmean="sd",
                    jitter=0.3,
                    pointpos=-1.8,
                    boxpoints="suspectedoutliers",
                )
            )

        fig.update_layout(
            **_base_layout(
                height=480,
                title_text="<b>Boxplots — Outlier & Spread View</b>",
                title_font_size=15,
                showlegend=False,
                xaxis=dict(tickangle=-35),
            )
        )
        return fig

    # ─────────────────────────────────────────────────────────────
    # BAR CHARTS (CATEGORICAL)
    # ─────────────────────────────────────────────────────────────

    def bar_chart_grid(
        self,
        df: pd.DataFrame,
        cat_cols: List[str],
        top_n: int = 12,
        max_plots: int = 9,
    ) -> go.Figure:
        """Grid of bar charts for categorical columns."""
        cols = [c for c in cat_cols if c in df.columns][:max_plots]
        if not cols:
            return self._empty("No categorical columns available.")

        n_cols = min(3, len(cols))
        n_rows = (len(cols) + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=cols,
            vertical_spacing=0.10,
            horizontal_spacing=0.06,
        )

        for idx, col in enumerate(cols):
            r, c = divmod(idx, n_cols)
            vc = df[col].value_counts().head(top_n)
            if len(vc) == 0:
                continue
            fig.add_trace(
                go.Bar(
                    x=vc.index.astype(str),
                    y=vc.values,
                    marker_color=_PALETTE[idx % len(_PALETTE)],
                    showlegend=False,
                    name=col,
                    text=vc.values,
                    textposition="auto",
                ),
                row=r + 1, col=c + 1,
            )

        fig.update_layout(
            **_base_layout(
                height=360 * n_rows,
                title_text=f"<b>Top {top_n} Values — Categorical Columns</b>",
                title_font_size=15,
                showlegend=False,
            )
        )
        return fig

    # ─────────────────────────────────────────────────────────────
    # TIME SERIES
    # ─────────────────────────────────────────────────────────────

    def time_series_grid(
        self,
        df: pd.DataFrame,
        dt_col: str,
        numeric_cols: List[str],
        max_series: int = 4,
    ) -> go.Figure:
        """Multi-panel time series for up to max_series numeric columns."""
        valid_num = [c for c in numeric_cols if c in df.columns][:max_series]
        if not valid_num or dt_col not in df.columns:
            return self._empty("Datetime or numeric columns not available.")

        n = len(valid_num)
        fig = make_subplots(
            rows=n, cols=1,
            subplot_titles=[f"{c} over time" for c in valid_num],
            shared_xaxes=True,
            vertical_spacing=0.06,
        )

        for i, col in enumerate(valid_num):
            tmp = df[[dt_col, col]].dropna().sort_values(dt_col)
            fig.add_trace(
                go.Scatter(
                    x=tmp[dt_col],
                    y=tmp[col],
                    mode="lines",
                    name=col,
                    line=dict(color=_PALETTE[i % len(_PALETTE)], width=2),
                    fill="tozeroy",
                    fillcolor=f"rgba({_hex_to_rgb(_PALETTE[i % len(_PALETTE)])},0.06)",
                ),
                row=i + 1, col=1,
            )

        fig.update_layout(
            **_base_layout(
                height=280 * n,
                title_text=f"<b>Time Series (x={dt_col})</b>",
                title_font_size=15,
                showlegend=True,
            )
        )
        return fig

    # ─────────────────────────────────────────────────────────────
    # CORRELATION HEATMAP
    # ─────────────────────────────────────────────────────────────

    def correlation_heatmap(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        method: str = "pearson",
    ) -> go.Figure:
        """Triangular correlation heatmap with annotation."""
        cols = [c for c in numeric_cols if c in df.columns]
        if len(cols) < 2:
            return self._empty("Need ≥ 2 numeric columns for correlation.")

        corr = df[cols].corr(method=method).round(2)
        mask = np.tril(np.ones_like(corr, dtype=bool))
        z    = corr.where(mask).values

        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=cols,
                y=cols,
                colorscale="RdBu_r",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=np.where(mask, corr.round(2).values.astype(str), ""),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False,
                colorbar=dict(title="r", thickness=14, len=0.75),
            )
        )

        fig.update_layout(
            **_base_layout(
                height=max(420, len(cols) * 46 + 120),
                title_text=f"<b>Correlation Heatmap ({method.title()})</b>",
                title_font_size=15,
                xaxis=dict(tickangle=-40),
            )
        )
        return fig

    # ─────────────────────────────────────────────────────────────
    # MISSING VALUE VISUALIZATIONS
    # ─────────────────────────────────────────────────────────────

    def missing_bar(self, missing_df: pd.DataFrame) -> go.Figure:
        """Horizontal bar chart of missing % per column."""
        if missing_df.empty:
            return self._empty("🎉 No missing values found in this dataset!")

        colors = [
            "#ef4444" if p > 50
            else "#f97316" if p > 20
            else "#f59e0b" if p > 5
            else "#10b981"
            for p in missing_df["Missing %"]
        ]

        fig = go.Figure(
            go.Bar(
                x=missing_df["Missing %"],
                y=missing_df["Column"],
                orientation="h",
                marker_color=colors,
                text=missing_df["Missing %"].apply(lambda v: f"{v:.1f}%"),
                textposition="auto",
            )
        )
        fig.add_vline(x=5,  line_dash="dash", line_color="#f59e0b",
                      annotation_text="5%",  annotation_position="top right")
        fig.add_vline(x=20, line_dash="dash", line_color="#f97316",
                      annotation_text="20%", annotation_position="top right")
        fig.add_vline(x=50, line_dash="dash", line_color="#ef4444",
                      annotation_text="50%", annotation_position="top right")

        fig.update_layout(
            **_base_layout(
                height=max(320, len(missing_df) * 26 + 100),
                title_text="<b>Missing Values by Column</b>",
                title_font_size=15,
                xaxis_title="Missing %",
                yaxis=dict(autorange="reversed"),
            )
        )
        return fig

    def missing_pattern_heatmap(
        self, df: pd.DataFrame, max_rows: int = 500
    ) -> go.Figure:
        """Binary heatmap showing where values are missing (red) vs present (dark)."""
        sample       = df.head(max_rows)
        miss_matrix  = sample.isna().astype(int)

        fig = go.Figure(
            go.Heatmap(
                z=miss_matrix.values.T,
                x=list(range(len(sample))),
                y=sample.columns.tolist(),
                colorscale=[[0, "#1a2744"], [1, "#ef4444"]],
                showscale=False,
                hovertemplate="Row: %{x}<br>Column: %{y}<br>Missing: %{z}<extra></extra>",
            )
        )
        fig.update_layout(
            **_base_layout(
                height=max(300, len(df.columns) * 22 + 100),
                title_text=f"<b>Missing Pattern — first {min(max_rows, len(df))} rows (red = missing)</b>",
                title_font_size=14,
                xaxis_title="Row index",
            )
        )
        return fig

    # ─────────────────────────────────────────────────────────────
    # OUTLIER CHART
    # ─────────────────────────────────────────────────────────────

    def outlier_comparison(self, outlier_df: pd.DataFrame) -> go.Figure:
        """Grouped bar comparing IQR vs Z-score outlier % per column."""
        if outlier_df.empty:
            return self._empty("No outlier data available.")

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="IQR (1.5×)",
                x=outlier_df["Column"],
                y=outlier_df["IQR Outliers %"],
                marker_color=_CYAN,
                text=outlier_df["IQR Outliers %"].apply(lambda v: f"{v:.1f}%"),
                textposition="auto",
            )
        )
        fig.add_trace(
            go.Bar(
                name="Z-Score (>3σ)",
                x=outlier_df["Column"],
                y=outlier_df["Z-Score Outliers %"],
                marker_color=_CORAL,
                text=outlier_df["Z-Score Outliers %"].apply(lambda v: f"{v:.1f}%"),
                textposition="auto",
            )
        )
        fig.update_layout(
            **_base_layout(
                height=440,
                title_text="<b>Outlier % by Column</b>",
                title_font_size=15,
                barmode="group",
                xaxis=dict(tickangle=-35),
                legend=dict(bgcolor="rgba(0,0,0,0.3)"),
            )
        )
        return fig

    # ─────────────────────────────────────────────────────────────
    # PAIRPLOT / SCATTER MATRIX
    # ─────────────────────────────────────────────────────────────

    def pairplot(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        max_cols: int = 5,
    ) -> go.Figure:
        """Scatter matrix for up to max_cols numeric columns."""
        cols = [c for c in numeric_cols if c in df.columns][:max_cols]
        if len(cols) < 2:
            return self._empty("Need ≥ 2 numeric columns for scatter matrix.")

        subset = df[cols].dropna()
        fig    = px.scatter_matrix(
            subset,
            dimensions=cols,
            opacity=0.45,
            color_discrete_sequence=[_CYAN],
        )
        fig.update_traces(
            diagonal_visible=True,
            showupperhalf=False,
            marker=dict(size=3),
        )
        fig.update_layout(
            **_base_layout(
                height=600,
                title_text="<b>Scatter Matrix (Pairplot)</b>",
                title_font_size=15,
            )
        )
        return fig

    # ─────────────────────────────────────────────────────────────
    # SINGLE COLUMN DEEP-DIVE
    # ─────────────────────────────────────────────────────────────

    def single_column_deep_dive(
        self, df: pd.DataFrame, col: str
    ) -> go.Figure:
        """Histogram + boxplot side-by-side for one numeric column."""
        if col not in df.columns:
            return self._empty(f"Column '{col}' not found.")
        s = df[col].dropna()
        if len(s) == 0:
            return self._empty(f"'{col}' has no non-null values.")

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[f"Histogram — {col}", f"Box Plot — {col}"],
        )
        fig.add_trace(
            go.Histogram(
                x=s, nbinsx=40,
                marker_color=_CYAN, opacity=0.80, name="Histogram",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Box(
                y=s, boxmean=True,
                marker_color=_CORAL, name="Boxplot",
                boxpoints="suspectedoutliers",
            ),
            row=1, col=2,
        )
        fig.update_layout(
            **_base_layout(height=420, showlegend=False)
        )
        return fig

    def single_cat_chart(
        self, df: pd.DataFrame, col: str, top_n: int = 20
    ) -> go.Figure:
        """Horizontal bar for one categorical column."""
        if col not in df.columns:
            return self._empty(f"Column '{col}' not found.")
        vc = df[col].value_counts().head(top_n)
        if len(vc) == 0:
            return self._empty(f"'{col}' has no non-null values.")

        pct = (vc / len(df) * 100).round(1)
        fig = go.Figure(
            go.Bar(
                x=vc.values,
                y=vc.index.astype(str),
                orientation="h",
                marker_color=_CYAN,
                text=[f"{v:,} ({p}%)" for v, p in zip(vc.values, pct)],
                textposition="auto",
            )
        )
        fig.update_layout(
            **_base_layout(
                height=max(320, len(vc) * 28 + 100),
                title_text=f"<b>{col} — Top {top_n} values</b>",
                title_font_size=14,
                yaxis=dict(autorange="reversed"),
            )
        )
        return fig

    # ─────────────────────────────────────────────────────────────
    # COLUMN TYPE DONUT
    # ─────────────────────────────────────────────────────────────

    def column_type_donut(self, col_types: dict) -> go.Figure:
        """Donut chart showing column type distribution."""
        label_map = {
            "numeric": "Numeric",
            "categorical": "Categorical",
            "datetime": "Datetime",
            "boolean": "Boolean",
            "high_cardinality": "High Cardinality",
            "text": "Free Text",
            "other": "Other / Empty",
        }
        color_map = {
            "Numeric": _CYAN,
            "Categorical": "#7c3aed",
            "Datetime": "#10b981",
            "Boolean": "#f59e0b",
            "High Cardinality": "#ec4899",
            "Free Text": "#6366f1",
            "Other / Empty": "#475569",
        }

        labels = []
        values = []
        colors = []
        for k, v in col_types.items():
            if v:
                lbl = label_map.get(k, k)
                labels.append(lbl)
                values.append(len(v))
                colors.append(color_map.get(lbl, "#94a3b8"))

        if not values:
            return self._empty("No column type data.")

        fig = go.Figure(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors),
                textinfo="label+percent",
                textfont=dict(size=11),
            )
        )
        _layout = _base_layout(
            height=340,
            title_text="<b>Column Types</b>",
            title_font_size=14,
            showlegend=False,
        )
        _layout["margin"] = dict(t=60, b=20, l=20, r=20)
        fig.update_layout(**_layout)
        return fig

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _empty(msg: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(
            text=msg, x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=14, color="#475569"),
        )
        fig.update_layout(
            template=_TMPL,
            paper_bgcolor=_BG,
            plot_bgcolor=_BG,
            height=280,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig


def _hex_to_rgb(hex_color: str) -> str:
    """Convert #rrggbb to 'r,g,b' string for rgba()."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"
