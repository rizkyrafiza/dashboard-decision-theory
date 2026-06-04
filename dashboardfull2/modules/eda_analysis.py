"""
modules/eda_analysis.py
─────────────────────────────────────────────
All statistical analysis: descriptive stats,
missing values, outliers, correlations.
Column-name-agnostic.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class EDAAnalyzer:
    """Runs statistical EDA on any DataFrame."""

    # ─────────────────────────────────────────────────────────────
    # DESCRIPTIVE STATISTICS
    # ─────────────────────────────────────────────────────────────

    def descriptive_stats(
        self, df: pd.DataFrame, numeric_cols: List[str]
    ) -> pd.DataFrame:
        """
        Extended describe() for numeric columns.
        Adds: skewness, kurtosis, CV%, IQR, missing count/%.
        """
        if not numeric_cols:
            return pd.DataFrame()

        valid = [c for c in numeric_cols if c in df.columns]
        if not valid:
            return pd.DataFrame()

        subset = df[valid]
        try:
            desc = subset.describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]).T
        except Exception:
            return pd.DataFrame()

        desc["skewness"]  = subset.skew().round(4)
        desc["kurtosis"]  = subset.kurtosis().round(4)
        desc["cv_%"]      = (subset.std() / subset.mean().abs() * 100).round(2)
        desc["iqr"]       = (subset.quantile(0.75) - subset.quantile(0.25)).round(4)
        desc["missing"]   = subset.isna().sum().astype(int)
        desc["missing_%"] = (subset.isna().mean() * 100).round(2)

        return desc.round(4)

    # ─────────────────────────────────────────────────────────────
    # MISSING VALUE ANALYSIS
    # ─────────────────────────────────────────────────────────────

    def missing_value_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Per-column missing-value breakdown.
        Sorted by missing %, descending.  Only columns with ≥1 missing value.
        """
        n        = len(df)
        miss_cnt = df.isna().sum()
        miss_pct = (miss_cnt / n * 100).round(2)

        result = pd.DataFrame(
            {
                "Column":         df.columns,
                "Missing Count":  miss_cnt.values.astype(int),
                "Missing %":      miss_pct.values,
                "Present Count":  (n - miss_cnt.values).astype(int),
                "Dtype":          df.dtypes.astype(str).values,
            }
        )

        result = (
            result[result["Missing Count"] > 0]
            .sort_values("Missing %", ascending=False)
            .reset_index(drop=True)
        )
        return result

    # ─────────────────────────────────────────────────────────────
    # OUTLIER DETECTION
    # ─────────────────────────────────────────────────────────────

    def outlier_detection(
        self, df: pd.DataFrame, numeric_cols: List[str]
    ) -> pd.DataFrame:
        """
        Outlier detection via IQR and Z-score (|z| > 3) methods.
        """
        if not numeric_cols:
            return pd.DataFrame()

        rows = []
        for col in numeric_cols:
            if col not in df.columns:
                continue
            s = df[col].dropna()
            if len(s) < 4:
                continue

            # IQR
            q1, q3  = s.quantile(0.25), s.quantile(0.75)
            iqr     = q3 - q1
            if iqr > 0:
                lo, hi      = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                iqr_out     = int(((s < lo) | (s > hi)).sum())
            else:
                lo, hi      = float(q1), float(q3)
                iqr_out     = 0

            # Z-score
            if s.std() > 0:
                z_out = int((np.abs(stats.zscore(s)) > 3).sum())
            else:
                z_out = 0

            rows.append(
                {
                    "Column":              col,
                    "IQR Outliers":        iqr_out,
                    "IQR Outliers %":      round(iqr_out / len(s) * 100, 2),
                    "Z-Score Outliers":    z_out,
                    "Z-Score Outliers %":  round(z_out / len(s) * 100, 2),
                    "Lower Bound (IQR)":   round(lo, 4),
                    "Upper Bound (IQR)":   round(hi, 4),
                    "Min":                 round(float(s.min()), 4),
                    "Max":                 round(float(s.max()), 4),
                }
            )

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ─────────────────────────────────────────────────────────────
    # CORRELATION ANALYSIS
    # ─────────────────────────────────────────────────────────────

    def correlation_analysis(
        self, df: pd.DataFrame, numeric_cols: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return (Pearson, Spearman) correlation matrices.
        """
        if len(numeric_cols) < 2:
            return pd.DataFrame(), pd.DataFrame()

        valid  = [c for c in numeric_cols if c in df.columns]
        subset = df[valid].dropna(how="all")

        pearson  = pd.DataFrame()
        spearman = pd.DataFrame()

        try:
            pearson = subset.corr(method="pearson").round(3)
        except Exception:
            pass
        try:
            spearman = subset.corr(method="spearman").round(3)
        except Exception:
            pass

        return pearson, spearman

    def get_strong_correlations(
        self, corr_matrix: pd.DataFrame, threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Extract pairs with |correlation| ≥ threshold from a
        correlation matrix (lower triangle only).
        """
        if corr_matrix.empty:
            return pd.DataFrame()

        pairs = []
        cols  = corr_matrix.columns.tolist()

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr_matrix.iloc[i, j]
                if abs(val) >= threshold:
                    pairs.append(
                        {
                            "Variable 1":  cols[i],
                            "Variable 2":  cols[j],
                            "Correlation": round(val, 3),
                            "Strength":    self._corr_label(abs(val)),
                            "Direction":   "↑ Positive" if val > 0 else "↓ Negative",
                        }
                    )

        if not pairs:
            return pd.DataFrame()
        return (
            pd.DataFrame(pairs)
            .sort_values("Correlation", key=abs, ascending=False)
            .reset_index(drop=True)
        )

    # ─────────────────────────────────────────────────────────────
    # CATEGORICAL ANALYSIS
    # ─────────────────────────────────────────────────────────────

    def categorical_analysis(
        self,
        df: pd.DataFrame,
        cat_cols: List[str],
        top_n: int = 15,
    ) -> Dict[str, pd.DataFrame]:
        """Return {col → DataFrame(Value, Count, Percentage)} for each categorical column."""
        result = {}
        for col in cat_cols:
            if col not in df.columns:
                continue
            vc  = df[col].value_counts().head(top_n)
            pct = df[col].value_counts(normalize=True).head(top_n) * 100
            result[col] = pd.DataFrame(
                {
                    "Value":      vc.index.astype(str),
                    "Count":      vc.values.astype(int),
                    "Percentage": pct.values.round(2),
                }
            )
        return result

    # ─────────────────────────────────────────────────────────────
    # DATASET SUMMARY
    # ─────────────────────────────────────────────────────────────

    def dataset_summary(
        self, df: pd.DataFrame, col_types: Dict[str, List[str]]
    ) -> Dict:
        """High-level summary dict for the Overview tab."""
        complete_rows = int(df.dropna().shape[0])
        return {
            "total_rows":       int(len(df)),
            "total_cols":       int(len(df.columns)),
            "numeric_cols":     len(col_types.get("numeric", [])),
            "categorical_cols": len(col_types.get("categorical", [])),
            "datetime_cols":    len(col_types.get("datetime", [])),
            "boolean_cols":     len(col_types.get("boolean", [])),
            "hc_cols":          len(col_types.get("high_cardinality", [])),
            "text_cols":        len(col_types.get("text", [])),
            "missing_pct":      round(df.isna().sum().sum() / max(df.size, 1) * 100, 2),
            "duplicate_pct":    round(df.duplicated().sum() / max(len(df), 1) * 100, 2),
            "complete_rows":    complete_rows,
            "complete_rows_pct": round(complete_rows / max(len(df), 1) * 100, 2),
        }

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _corr_label(val: float) -> str:
        if val >= 0.90: return "Very Strong"
        if val >= 0.70: return "Strong"
        if val >= 0.50: return "Moderate"
        if val >= 0.30: return "Weak"
        return "Very Weak"
