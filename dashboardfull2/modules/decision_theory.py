"""
modules/decision_theory.py
────────────────────────────────────────────────────────────────
Modul Analisis Teori Pengambilan Keputusan
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Implementasi teori pengambilan keputusan klasik:
  1. Nilai Ekspektasi (Expected Value / EV)
  2. Analisis Risiko & Utilitas (Risk-Utility Profile)
  3. Matriks Keputusan (Decision Matrix)
  4. Kriteria Keputusan: Maximax · Maximin · Minimax Regret ·
                         Laplace · Hurwicz · Expected Value
  5. Analisis Dominansi (Dominance Analysis)
  6. Analisis Multi-Kriteria / MCDA (Weighted Scoring)
  7. Analisis Sensitivitas (Sensitivity Analysis)
  8. Analisis Pareto (Pareto Frontier)
  9. Profil Toleransi Risiko (Risk Tolerance Profile)

Tidak menggunakan nama kolom hardcoded.
Semua analisis digerakkan oleh statistik & tipe data.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    try:
        v = float(val)
        return None if (np.isnan(v) or np.isinf(v)) else v
    except Exception:
        return None


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalisasi ke [0, 1]."""
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / (mx - mn)


# ─────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────

class DecisionTheoryAnalyzer:
    """
    Analisis teori pengambilan keputusan untuk dataset sembarang.
    Semua metode bersifat generik — tidak ada asumsi nama kolom.
    """

    # ══════════════════════════════════════════════════════════════
    # 1. NILAI EKSPEKTASI & PROFIL RISIKO
    # ══════════════════════════════════════════════════════════════

    def expected_value_profile(
        self, df: pd.DataFrame, numeric_cols: List[str]
    ) -> pd.DataFrame:
        """
        Untuk setiap kolom numerik, hitung:
        EV (mean), risiko (std), CV%, probabilitas > mean,
        probabilitas > 0, nilai skewness, kurtosis,
        dan kategorisasi profil risiko.
        """
        rows = []
        for col in numeric_cols:
            if col not in df.columns:
                continue
            s = df[col].dropna()
            if len(s) < 2:
                continue

            ev   = float(s.mean())
            std  = float(s.std())
            cv   = abs(std / ev * 100) if ev != 0 else np.inf
            skew = float(s.skew())
            kurt = float(s.kurtosis())
            p_gt_mean = float((s > ev).mean() * 100)
            p_gt_zero = float((s > 0).mean() * 100) if s.min() <= 0 else 100.0

            # Kategori risiko berdasarkan CV
            if cv <= 15:
                risk_cat = "🟢 Rendah"
            elif cv <= 30:
                risk_cat = "🟡 Sedang"
            elif cv <= 60:
                risk_cat = "🟠 Tinggi"
            else:
                risk_cat = "🔴 Sangat Tinggi"

            # Distribusi: normal, right-skewed, left-skewed
            if abs(skew) < 0.5:
                dist_type = "Simetris"
            elif skew > 0.5:
                dist_type = "Miring Kanan (Right-skewed)"
            else:
                dist_type = "Miring Kiri (Left-skewed)"

            rows.append({
                "Kolom":                col,
                "Nilai Ekspektasi (EV)": round(ev, 4),
                "Simpangan Baku (σ)":    round(std, 4),
                "Koef. Variasi (%)":     round(cv, 2) if not np.isinf(cv) else "∞",
                "Skewness":              round(skew, 4),
                "Kurtosis":              round(kurt, 4),
                "P(X > EV) %":           round(p_gt_mean, 2),
                "P(X > 0) %":            round(p_gt_zero, 2),
                "Tipe Distribusi":        dist_type,
                "Kategori Risiko":        risk_cat,
            })

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ══════════════════════════════════════════════════════════════
    # 2. MATRIKS KEPUTUSAN
    # ══════════════════════════════════════════════════════════════

    def build_decision_matrix(
        self,
        df: pd.DataFrame,
        state_col: str,        # kolom kategorik = "kondisi alam / skenario"
        outcome_cols: List[str],  # kolom numerik = "hasil / payoff"
        agg: str = "mean",
    ) -> pd.DataFrame:
        """
        Bangun matriks keputusan di mana:
          - Baris  = nilai unik state_col (kondisi alam / skenario)
          - Kolom  = outcome_cols (alternatif / metrik keputusan)
          - Nilai  = agregat (mean/median/sum) payoff per kondisi
        """
        if state_col not in df.columns:
            return pd.DataFrame()
        valid_oc = [c for c in outcome_cols if c in df.columns]
        if not valid_oc:
            return pd.DataFrame()

        agg_fn = {"mean": "mean", "median": "median", "sum": "sum"}.get(agg, "mean")
        matrix = (
            df[[state_col] + valid_oc]
            .groupby(state_col, observed=True)
            .agg(agg_fn)
            .round(4)
        )
        return matrix

    # ══════════════════════════════════════════════════════════════
    # 3. KRITERIA KEPUTUSAN KLASIK
    # ══════════════════════════════════════════════════════════════

    def apply_decision_criteria(
        self,
        matrix: pd.DataFrame,
        probabilities: Optional[Dict[str, float]] = None,
        hurwicz_alpha: float = 0.5,
        maximize: bool = True,
    ) -> pd.DataFrame:
        """
        Terapkan 6 kriteria keputusan klasik pada matriks keputusan.

        Baris matriks = kondisi alam (states of nature)
        Kolom matriks = alternatif keputusan

        Kriteria:
          1. Maximax / Minimin  — optimis
          2. Maximin / Minimax  — pesimis (konservatif)
          3. Minimax Regret     — minimasi penyesalan
          4. Laplace            — rata-rata tanpa bobot
          5. Hurwicz            — α·max + (1-α)·min
          6. Expected Value     — EV berbobot probabilitas
        """
        if matrix.empty:
            return pd.DataFrame()

        alt_cols = matrix.columns.tolist()
        rows     = []

        for alt in alt_cols:
            col_vals = matrix[alt].values.astype(float)

            # ── Maximax / Minimin
            best_val  = col_vals.max() if maximize else col_vals.min()

            # ── Maximin / Minimax
            worst_val = col_vals.min() if maximize else col_vals.max()

            # ── Laplace (equal probability)
            laplace_val = col_vals.mean()

            # ── Hurwicz
            hurwicz_val = (
                hurwicz_alpha * best_val + (1 - hurwicz_alpha) * worst_val
            )

            # ── Expected Value (berbobot prob)
            if probabilities:
                ev_val = sum(
                    col_vals[i] * probabilities.get(str(s), 1 / len(col_vals))
                    for i, s in enumerate(matrix.index)
                )
            else:
                ev_val = laplace_val

            rows.append({
                "Alternatif":           alt,
                "Maximax (Terbaik)":    round(best_val, 4),
                "Maximin (Terburuk)":   round(worst_val, 4),
                "Laplace (Rata-rata)":  round(laplace_val, 4),
                f"Hurwicz (α={hurwicz_alpha})": round(hurwicz_val, 4),
                "Nilai Ekspektasi (EV)": round(ev_val, 4),
            })

        crit_df = pd.DataFrame(rows)

        # ── Minimax Regret (dihitung dari matriks penuh)
        regret_matrix = self._regret_matrix(matrix, maximize)
        if not regret_matrix.empty:
            max_regret = regret_matrix.max(axis=0)
            crit_df["Minimax Regret"] = [
                round(float(max_regret[alt]), 4) if alt in max_regret.index else np.nan
                for alt in crit_df["Alternatif"]
            ]

        # ── Rekomendasi per kriteria
        crit_df = self._add_recommendations(crit_df, maximize)
        return crit_df

    def _regret_matrix(self, matrix: pd.DataFrame, maximize: bool) -> pd.DataFrame:
        """Hitung matriks penyesalan (opportunity loss)."""
        if matrix.empty:
            return pd.DataFrame()
        ref = matrix.max(axis=1) if maximize else matrix.min(axis=1)
        regret = matrix.copy().astype(float)
        for col in regret.columns:
            regret[col] = (ref - matrix[col]).abs()
        return regret.round(4)

    def get_regret_matrix(self, matrix: pd.DataFrame, maximize: bool = True) -> pd.DataFrame:
        return self._regret_matrix(matrix, maximize)

    def _add_recommendations(self, crit_df: pd.DataFrame, maximize: bool) -> pd.DataFrame:
        """Tandai alternatif terbaik per kriteria."""
        criteria_map = {
            "Maximax (Terbaik)":    maximize,
            "Maximin (Terburuk)":   maximize,
            "Laplace (Rata-rata)":  maximize,
            "Nilai Ekspektasi (EV)": maximize,
            "Minimax Regret":       False,   # selalu minimasi regret
        }
        for k in crit_df.columns:
            if k.startswith("Hurwicz"):
                criteria_map[k] = maximize

        rec_rows = []
        for _, row in crit_df.iterrows():
            rec_rows.append(row["Alternatif"])

        best_per_crit = {}
        for crit, is_max in criteria_map.items():
            if crit not in crit_df.columns:
                continue
            try:
                idx = (
                    crit_df[crit].idxmax() if is_max else crit_df[crit].idxmin()
                )
                best_per_crit[crit] = crit_df.loc[idx, "Alternatif"]
            except Exception:
                pass

        # Tambah kolom tanda ★
        for crit, winner in best_per_crit.items():
            if crit in crit_df.columns:
                crit_df[crit] = crit_df.apply(
                    lambda r: f"★ {r[crit]}" if r["Alternatif"] == winner else r[crit],
                    axis=1,
                )
        return crit_df

    # ══════════════════════════════════════════════════════════════
    # 4. ANALISIS DOMINANSI
    # ══════════════════════════════════════════════════════════════

    def dominance_analysis(
        self, df: pd.DataFrame, numeric_cols: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Bandingkan semua pasangan kolom numerik.
        Kolom A mendominasi kolom B jika A ≥ B di semua baris.
        Kembalikan:
          - tabel dominansi (pasangan)
          - matriks dominansi (biner)
        """
        valid = [c for c in numeric_cols if c in df.columns]
        if len(valid) < 2:
            return pd.DataFrame(), pd.DataFrame()

        sub   = df[valid].dropna()
        n     = len(valid)
        pairs = []
        matrix_data = np.zeros((n, n), dtype=int)

        for i, a in enumerate(valid):
            for j, b in enumerate(valid):
                if i == j:
                    continue
                diff = sub[a].values - sub[b].values
                if np.all(diff >= 0):
                    rel   = "Dominasi Penuh"
                    dom   = 1
                elif np.all(diff <= 0):
                    rel   = "Didominasi Penuh"
                    dom   = 0
                elif np.mean(diff >= 0) >= 0.75:
                    rel   = "Dominasi Parsial (≥75%)"
                    dom   = 1
                else:
                    rel   = "Tidak Ada Dominansi"
                    dom   = 0
                matrix_data[i, j] = dom
                if i < j:
                    pairs.append({
                        "Alternatif A": a,
                        "Alternatif B": b,
                        "Relasi":       rel,
                        "A mendominasi B": "✅ Ya" if dom else "❌ Tidak",
                        "% Baris A ≥ B": round(float(np.mean(diff >= 0)) * 100, 1),
                    })

        dom_matrix = pd.DataFrame(matrix_data, index=valid, columns=valid)
        return pd.DataFrame(pairs), dom_matrix

    # ══════════════════════════════════════════════════════════════
    # 5. ANALISIS MULTI-KRITERIA (MCDA – Weighted Scoring)
    # ══════════════════════════════════════════════════════════════

    def mcda_weighted_scoring(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        weights: Optional[Dict[str, float]] = None,
        maximize_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Multi-Criteria Decision Analysis (MCDA)
        menggunakan metode Weighted Scoring.
        """

        valid = [c for c in numeric_cols if c in df.columns]

        if not valid:
            return pd.DataFrame()

        sub = df[valid].copy().dropna()

        if sub.empty:
            return pd.DataFrame()

        if maximize_cols is None:
            maximize_cols = valid

        # Normalisasi
        norm = sub.copy()

        for col in valid:
            arr = sub[col].astype(float).values

            if col in maximize_cols:
                norm[col] = _normalize(arr)
            else:
                norm[col] = 1 - _normalize(arr)

        # Bobot
        if weights is None:
            w = {c: 1.0 / len(valid) for c in valid}
        else:
            total = sum(weights.get(c, 1.0) for c in valid)

            if total == 0:
                w = {c: 1.0 / len(valid) for c in valid}
            else:
                w = {
                    c: weights.get(c, 1.0) / total
                    for c in valid
                }

        # Skor MCDA
        score = pd.Series(0.0, index=sub.index)

        for col in valid:
            score += norm[col] * w[col]

        result = sub.copy()

        result["Skor MCDA"] = score.round(4)

        result["Peringkat"] = (
            result["Skor MCDA"]
            .rank(method="dense", ascending=False)
            .astype(int)
        )

        return (
            result
            .sort_values("Skor MCDA", ascending=False)
            .reset_index(drop=True)
        )

    # ══════════════════════════════════════════════════════════════
    # 6. ANALISIS SENSITIVITAS
    # ══════════════════════════════════════════════════════════════

    def sensitivity_analysis(
        self,
        df: pd.DataFrame,
        target_col: str,
        predictor_cols: List[str],
        perturbation_pct: float = 10.0,
    ) -> pd.DataFrame:
        """
        Ukur sensitivitas kolom target terhadap perubahan ±perturbation_pct%
        pada setiap kolom prediktor.

        Output: tabel elastisitas & dampak per prediktor.
        """
        if target_col not in df.columns:
            return pd.DataFrame()
        valid = [c for c in predictor_cols if c in df.columns and c != target_col]
        if not valid:
            return pd.DataFrame()

        sub       = df[[target_col] + valid].dropna()
        base_mean = float(sub[target_col].mean())
        if base_mean == 0:
            return pd.DataFrame()

        rows = []
        for col in valid:
            col_mean = float(sub[col].mean())
            if col_mean == 0:
                continue

            delta    = col_mean * perturbation_pct / 100
            perturb  = sub.copy()
            perturb[col] = sub[col] + delta

            # Regresi linear sederhana untuk estimasi dampak
            try:
                slope, intercept, r, p_val, _ = stats.linregress(sub[col], sub[target_col])
            except Exception:
                slope, r, p_val = 0.0, 0.0, 1.0

            pred_base = slope * col_mean + intercept
            pred_new  = slope * (col_mean + delta) + intercept
            impact    = pred_new - pred_base
            elasticity = (impact / base_mean) / (delta / col_mean) if col_mean != 0 else 0

            rows.append({
                "Prediktor":               col,
                "Gangguan ±%":             perturbation_pct,
                "Korelasi (r)":            round(float(r), 4),
                "p-value":                 round(float(p_val), 4),
                "Slope Regresi":           round(float(slope), 6),
                "Dampak pada Target":      round(float(impact), 4),
                "Dampak %":                round(float(impact / base_mean * 100), 4),
                "Elastisitas":             round(float(elasticity), 4),
                "Signifikan (p<0.05)":     "✅ Ya" if p_val < 0.05 else "❌ Tidak",
                "Tingkat Sensitivitas":    self._sens_label(abs(elasticity)),
            })

        if not rows:
            return pd.DataFrame()
        return (
            pd.DataFrame(rows)
            .sort_values("Elastisitas", key=abs, ascending=False)
            .reset_index(drop=True)
        )

    @staticmethod
    def _sens_label(e: float) -> str:
        if e >= 2.0:  return "🔴 Sangat Sensitif"
        if e >= 1.0:  return "🟠 Sensitif (Elastis)"
        if e >= 0.5:  return "🟡 Sedang"
        return              "🟢 Tidak Sensitif"

    # ══════════════════════════════════════════════════════════════
    # 7. ANALISIS PARETO (FRONTIER)
    # ══════════════════════════════════════════════════════════════

    def pareto_analysis(
        self,
        df: pd.DataFrame,
        x_col: str,   # sumbu X = risiko / biaya
        y_col: str,   # sumbu Y = manfaat / return
        label_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Identifikasi titik-titik pada Pareto frontier (non-dominated).
        Setiap titik mewakili satu baris dataset (atau agregat per kategori).
        """
        if x_col not in df.columns or y_col not in df.columns:
            return pd.DataFrame()

        sub = df[[c for c in [label_col, x_col, y_col] if c and c in df.columns]].dropna()
        if len(sub) < 2:
            return pd.DataFrame()

        x = sub[x_col].values.astype(float)
        y = sub[y_col].values.astype(float)

        # Pareto: minimal x, maksimal y
        dominated = np.zeros(len(sub), dtype=bool)
        for i in range(len(sub)):
            for j in range(len(sub)):
                if i == j:
                    continue
                if x[j] <= x[i] and y[j] >= y[i] and (x[j] < x[i] or y[j] > y[i]):
                    dominated[i] = True
                    break

        result = sub.copy()
        result["Status Pareto"] = np.where(~dominated, "✅ Frontier", "❌ Terdominasi")
        result["Skor Efisiensi"] = (
            _normalize(y) - _normalize(x) + 1
        ).round(4)
        return result.sort_values("Skor Efisiensi", ascending=False).reset_index(drop=True)

    # ══════════════════════════════════════════════════════════════
    # 8. PROFIL TOLERANSI RISIKO
    # ══════════════════════════════════════════════════════════════

    def risk_tolerance_profile(
        self, df: pd.DataFrame, numeric_cols: List[str]
    ) -> pd.DataFrame:
        """
        Kategorikan setiap kolom numerik ke dalam tipe utilitas:
          - Risk Averse (menghindari risiko): CV rendah, distribusi simetris
          - Risk Neutral (netral risiko)
          - Risk Seeking (mencari risiko): CV tinggi, skewed kanan
        Dan hitung Certainty Equivalent (CE) estimatif.
        """
        rows = []
        for col in numeric_cols:
            if col not in df.columns:
                continue
            s = df[col].dropna()
            if len(s) < 4:
                continue

            ev   = float(s.mean())
            std  = float(s.std())
            skew = float(s.skew())
            cv   = abs(std / ev * 100) if ev != 0 else 999

            # Certainty Equivalent (approx Arrow-Pratt)
            variance = std ** 2
            # CE ≈ EV - (risk_aversion * variance / 2)
            # Gunakan CV sebagai proxy risk aversion
            ra        = min(cv / 100, 1.0)   # normalisasi
            ce        = ev - (ra * variance / (2 * abs(ev))) if ev != 0 else ev
            risk_prem = ev - ce

            # Tipe utilitas
            if cv <= 20 and abs(skew) < 0.5:
                utility_type = "⚖️ Risk Averse (Menghindari Risiko)"
                pref_desc    = "Lebih menyukai hasil pasti daripada hasil tak pasti"
            elif cv >= 50 or (skew > 1.0):
                utility_type = "🎲 Risk Seeking (Mencari Risiko)"
                pref_desc    = "Bersedia menanggung risiko tinggi demi hasil besar"
            else:
                utility_type = "⚡ Risk Neutral (Netral Risiko)"
                pref_desc    = "Keputusan semata berdasarkan nilai ekspektasi"

            # Value at Risk 5% (VaR)
            var_5pct = float(np.percentile(s, 5))

            rows.append({
                "Kolom":                  col,
                "EV":                     round(ev, 4),
                "Std Dev (σ)":            round(std, 4),
                "CV %":                   round(cv, 2),
                "Certainty Equivalent":   round(ce, 4),
                "Risk Premium":           round(risk_prem, 4),
                "VaR 5%":                 round(var_5pct, 4),
                "Tipe Utilitas":          utility_type,
                "Deskripsi Preferensi":   pref_desc,
            })

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ══════════════════════════════════════════════════════════════
    # 9. ANALISIS SKENARIO (What-If)
    # ══════════════════════════════════════════════════════════════

    def scenario_analysis(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        scenarios: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Simulasi 3 skenario (Pesimis / Realistis / Optimis) berdasarkan
        persentil statistik kolom numerik.
        """
        if not numeric_cols:
            return pd.DataFrame()

        if scenarios is None:
            scenarios = {"Pesimis (P10)": 0.10, "Realistis (P50)": 0.50, "Optimis (P90)": 0.90}

        rows = []
        for col in numeric_cols:
            if col not in df.columns:
                continue
            s = df[col].dropna()
            if len(s) < 4:
                continue

            ev = float(s.mean())
            row = {"Kolom": col, "Nilai Ekspektasi": round(ev, 4)}
            for label, pct in scenarios.items():
                pval = float(np.percentile(s, pct * 100))
                dev  = pval - ev
                row[label] = round(pval, 4)
                row[f"Deviasi {label.split('(')[0].strip()}"] = round(dev, 4)

            # Rentang skenario
            vals = [float(np.percentile(s, p * 100)) for p in scenarios.values()]
            row["Rentang Skenario"] = round(max(vals) - min(vals), 4)
            row["Rasio Optimis/Pesimis"] = round(vals[-1] / vals[0], 4) if vals[0] != 0 else "∞"
            rows.append(row)

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ══════════════════════════════════════════════════════════════
    # 10. RINGKASAN KEPUTUSAN
    # ══════════════════════════════════════════════════════════════

    def decision_summary(
        self,
        ev_profile: pd.DataFrame,
        risk_profile: pd.DataFrame,
        sensitivity: Optional[pd.DataFrame] = None,
    ) -> List[Dict]:
        """
        Hasilkan daftar temuan dan rekomendasi pengambilan keputusan
        berdasarkan semua analisis yang telah dijalankan.
        """
        findings = []

        if not ev_profile.empty and "Nilai Ekspektasi (EV)" in ev_profile.columns:
            # Kolom dengan EV tertinggi
            best_ev = ev_profile.loc[ev_profile["Nilai Ekspektasi (EV)"].idxmax()]
            findings.append({
                "ikon": "🏆",
                "judul": "Nilai Ekspektasi Tertinggi",
                "detail": (
                    f"**{best_ev['Kolom']}** memiliki nilai ekspektasi tertinggi "
                    f"sebesar **{best_ev['Nilai Ekspektasi (EV)']}** dengan risiko "
                    f"**{best_ev['Kategori Risiko']}**."
                ),
                "tipe": "success",
            })

            # Kolom paling berisiko
            numeric_cv = pd.to_numeric(ev_profile["Koef. Variasi (%)"], errors="coerce")
            if numeric_cv.notna().any():
                worst_risk = ev_profile.loc[numeric_cv.idxmax()]
                findings.append({
                    "ikon": "⚠️",
                    "judul": "Risiko Tertinggi",
                    "detail": (
                        f"**{worst_risk['Kolom']}** memiliki koefisien variasi "
                        f"**{worst_risk['Koef. Variasi (%)']}%** — perlu kehati-hatian "
                        f"dalam pengambilan keputusan berbasis variabel ini."
                    ),
                    "tipe": "warning",
                })

        if not risk_profile.empty and "Risk Premium" in risk_profile.columns:
            high_prem = risk_profile[pd.to_numeric(risk_profile["Risk Premium"], errors="coerce").abs() > 0]
            if not high_prem.empty:
                findings.append({
                    "ikon": "💰",
                    "judul": "Risk Premium Signifikan",
                    "detail": (
                        f"Ditemukan **{len(high_prem)} variabel** dengan risk premium "
                        f"signifikan. Pengambil keputusan yang risk averse perlu "
                        f"menyesuaikan certainty equivalent sebelum memutuskan."
                    ),
                    "tipe": "info",
                })

        if sensitivity is not None and not sensitivity.empty:
            if "Elastisitas" in sensitivity.columns:
                top_sens = sensitivity.iloc[0]
                findings.append({
                    "ikon": "📡",
                    "judul": "Variabel Paling Sensitif",
                    "detail": (
                        f"**{top_sens['Prediktor']}** adalah prediktor paling sensitif "
                        f"(elastisitas = **{top_sens['Elastisitas']}**). "
                        f"Perubahan kecil pada variabel ini berdampak besar pada keluaran."
                    ),
                    "tipe": "warning" if abs(float(top_sens["Elastisitas"])) > 1 else "info",
                })

        if not findings:
            findings.append({
                "ikon": "ℹ️",
                "judul": "Tidak Cukup Data",
                "detail": "Butuh minimal 2 kolom numerik untuk analisis keputusan.",
                "tipe": "info",
            })

        return findings
