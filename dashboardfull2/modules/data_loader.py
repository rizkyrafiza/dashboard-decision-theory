"""
modules/data_loader.py
─────────────────────────────────────────────
Handles loading of CSV / XLSX files and
automatic, heuristic-based column-type detection.
Zero hardcoded column names.
"""

import io
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class DataLoader:
    """
    Loads tabular data from uploaded files and classifies every
    column into one of seven semantic types without assuming any
    particular column names.
    """

    # Case-insensitive boolean token sets
    _BOOL_TRUE  = {"true",  "yes", "1", "1.0", "t", "y", "on"}
    _BOOL_FALSE = {"false", "no",  "0", "0.0", "f", "n", "off"}
    _BOOL_ALL   = _BOOL_TRUE | _BOOL_FALSE

    # Cardinality thresholds
    _CAT_MAX_UNIQUE_ABS   = 50   # absolute unique count cap for "categorical"
    _CAT_MAX_UNIQUE_RATIO = 0.5  # unique/total ratio cap
    _TEXT_MIN_AVG_LEN     = 40   # average token length above which → "text"

    # Month abbreviations used in datetime heuristic
    _MONTH_STRS = [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec",
    ]

    # ─────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────

    def load_file(self, uploaded_file) -> pd.DataFrame:
        """
        Entry point: accept a Streamlit UploadedFile and return a
        cleaned DataFrame.  Supports .csv, .xlsx, .xls.
        """
        fname = uploaded_file.name.lower()

        if fname.endswith(".csv"):
            df = self._load_csv(uploaded_file)
        elif fname.endswith((".xlsx", ".xls")):
            df = self._load_excel(uploaded_file)
        else:
            raise ValueError(
                f"Unsupported format '{uploaded_file.name}'.  "
                "Please upload a .csv or .xlsx file."
            )

        df.columns = [
            str(c).strip().replace("\n", " ").replace("\r", "")
            for c in df.columns
        ]
        df = df.dropna(how="all").reset_index(drop=True)
        df = df.loc[:, df.notna().any()]
        return df

    def detect_column_types(
        self, df: pd.DataFrame
    ) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
        """
        Classify every column.

        Returns
        -------
        types : dict
            Keys: 'numeric', 'categorical', 'datetime',
                  'boolean', 'high_cardinality', 'text', 'other'
        df_proc : pd.DataFrame
            Copy of df with datetime columns properly parsed.
        """
        types: Dict[str, List[str]] = {
            "numeric": [], "categorical": [], "datetime": [],
            "boolean": [], "high_cardinality": [], "text": [], "other": [],
        }
        df_proc = df.copy()

        for col in df.columns:
            series     = df[col]
            n_non_null = int(series.notna().sum())

            if n_non_null == 0:
                types["other"].append(col)
                continue

            if self._is_boolean(series):
                types["boolean"].append(col)
                continue

            if pd.api.types.is_numeric_dtype(series):
                types["numeric"].append(col)
                continue

            parsed_dt = self._try_datetime(series)
            if parsed_dt is not None:
                df_proc[col] = parsed_dt
                types["datetime"].append(col)
                continue

            # Numeric disguised as string ("$1,234", "12.5 %")
            num_guess = self._try_numeric_string(series, n_non_null)
            if num_guess is not None:
                df_proc[col] = num_guess
                types["numeric"].append(col)
                continue

            # Categorical vs high-cardinality vs free-text
            n_unique = int(series.nunique())
            ratio    = n_unique / max(len(series), 1)
            avg_len  = float(series.dropna().astype(str).str.len().mean())

            if n_unique <= self._CAT_MAX_UNIQUE_ABS or ratio <= self._CAT_MAX_UNIQUE_RATIO:
                types["categorical"].append(col)
            elif avg_len > self._TEXT_MIN_AVG_LEN:
                types["text"].append(col)
            else:
                types["high_cardinality"].append(col)

        return types, df_proc

    def get_basic_info(self, df: pd.DataFrame) -> Dict:
        """Return a flat dict of key dataset characteristics."""
        return {
            "n_rows":        int(len(df)),
            "n_cols":        int(len(df.columns)),
            "n_cells":       int(df.size),
            "memory_mb":     round(df.memory_usage(deep=True).sum() / 1024**2, 3),
            "n_duplicates":  int(df.duplicated().sum()),
            "dup_pct":       round(df.duplicated().sum() / max(len(df), 1) * 100, 2),
            "total_missing": int(df.isna().sum().sum()),
            "missing_pct":   round(df.isna().sum().sum() / max(df.size, 1) * 100, 2),
            "columns":       df.columns.tolist(),
        }

    # ─────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────

    def _load_csv(self, uploaded_file) -> pd.DataFrame:
        raw = uploaded_file.read()

        # Encoding detection: try common encodings in order
        used_enc = "utf-8"
        for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252", "iso-8859-1"):
            try:
                raw.decode(enc)
                used_enc = enc
                break
            except (UnicodeDecodeError, LookupError):
                continue

        # Delimiter detection from first line
        try:
            first_line = raw.decode(used_enc, errors="replace").split("\n")[0]
        except Exception:
            first_line = ""

        sep = ","
        best = first_line.count(",")
        for cand in (";", "\t", "|"):
            cnt = first_line.count(cand)
            if cnt > best:
                best = cnt
                sep  = cand

        try:
            return pd.read_csv(
                io.BytesIO(raw), encoding=used_enc, sep=sep,
                low_memory=False, on_bad_lines="skip",
            )
        except Exception:
            return pd.read_csv(
                io.BytesIO(raw), encoding="latin-1", sep=",",
                low_memory=False, on_bad_lines="skip",
            )

    def _load_excel(self, uploaded_file) -> pd.DataFrame:
        return pd.read_excel(uploaded_file, engine="openpyxl")

    def _is_boolean(self, series: pd.Series) -> bool:
        if series.dtype == bool:
            return True
        if pd.api.types.is_numeric_dtype(series):
            unique = set(series.dropna().unique())
            return unique.issubset({0, 1, 0.0, 1.0, True, False})
        unique_str = {str(v).strip().lower() for v in series.dropna().unique()}
        return bool(unique_str) and unique_str.issubset(self._BOOL_ALL)

    def _try_datetime(self, series: pd.Series) -> Optional[pd.Series]:
        if pd.api.types.is_numeric_dtype(series):
            return None
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return None
        joined = " ".join(sample.astype(str).tolist()).lower()
        has_date = (
            any(c in joined for c in ("/", "-", ":")) and
            any(c.isdigit() for c in joined)
        )
        has_month = any(m in joined for m in self._MONTH_STRS)
        if not (has_date or has_month):
            return None
        try:
            parsed  = pd.to_datetime(series, errors="coerce", dayfirst=False)
            n_valid = series.notna().sum()
            if parsed.notna().sum() / max(n_valid, 1) >= 0.70:
                return parsed
        except Exception:
            pass
        return None

    def _try_numeric_string(
        self, series: pd.Series, n_non_null: int
    ) -> Optional[pd.Series]:
        try:
            cleaned = (
                series.astype(str)
                      .str.strip()
                      .str.replace(r"[\$€£¥,\s%]", "", regex=True)
            )
            num = pd.to_numeric(cleaned, errors="coerce")
            if num.notna().sum() / max(n_non_null, 1) >= 0.85:
                return num
        except Exception:
            pass
        return None
