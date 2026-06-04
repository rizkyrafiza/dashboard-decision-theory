"""
modules/ai_insights.py
─────────────────────────────────────────────
Calls the Anthropic API with rich dataset metadata
and returns structured, actionable EDA insights.
Zero hardcoded column names; all analysis is
driven by statistics and dtypes.
"""

import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# JSON-safe serialisation helper
# ─────────────────────────────────────────────────────────────

class _SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):    return int(obj)
        if isinstance(obj, (np.floating,)):   return float(obj)
        if isinstance(obj, (np.bool_,)):      return bool(obj)
        if isinstance(obj, (np.ndarray,)):    return obj.tolist()
        if isinstance(obj, pd.Timestamp):     return str(obj)
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return super().default(obj)


# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a world-class Data Scientist specializing in automated Exploratory Data Analysis (EDA).

You will receive JSON metadata about a dataset — shape, column names, inferred types, descriptive statistics, cardinality, missing values, and correlations — and must produce comprehensive, actionable insights.

CRITICAL RULES:
1. Never assume a domain — derive it purely from the data characteristics.
2. Never assume column semantics from names alone; use statistics, dtypes, and cardinality.
3. Be specific: reference actual column names from the metadata.
4. Be concise but complete.

You MUST respond with a SINGLE valid JSON object with EXACTLY this schema (no markdown fences, no preamble):
{
  "domain": {
    "detected_domain": "string",
    "confidence": "high | medium | low",
    "reasoning": "string (≤80 words)"
  },
  "dataset_quality": {
    "score": 1-10,
    "label": "Poor | Fair | Good | Excellent",
    "strengths": ["string", ...],
    "concerns": ["string", ...]
  },
  "key_insights": [
    {
      "title": "string",
      "description": "string (≤120 words)",
      "type": "pattern | anomaly | relationship | distribution | data_quality",
      "importance": "high | medium | low",
      "affected_columns": ["string", ...]
    }
  ],
  "recommended_visualisations": [
    {
      "chart_type": "string",
      "columns": ["string", ...],
      "reason": "string (≤60 words)",
      "priority": "high | medium | low"
    }
  ],
  "data_quality_issues": [
    {
      "issue": "string",
      "severity": "high | medium | low",
      "affected_columns": ["string", ...],
      "recommendation": "string"
    }
  ],
  "business_questions": ["string", ...],
  "next_steps": ["string", ...]
}"""


class AIInsights:
    """
    Generates AI-powered dataset insights via the Anthropic API.
    Gracefully degrades if the API key is missing or a call fails.
    """

    MODEL = "claude-opus-4-6"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client  = None
        self._init_client()

    def _init_client(self):
        if not ANTHROPIC_AVAILABLE:
            return
        try:
            if self.api_key:
                self.client = anthropic.Anthropic(api_key=self.api_key)
            else:
                self.client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
        except Exception:
            self.client = None

    # ─────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self.client is not None

    def build_metadata(
        self,
        df: pd.DataFrame,
        col_types: Dict[str, List[str]],
        desc_stats: pd.DataFrame,
        missing_df: pd.DataFrame,
        strong_corrs: pd.DataFrame,
    ) -> str:
        """
        Assemble dataset metadata into a compact JSON string
        that will be sent to the LLM.
        """
        meta: Dict = {}

        # ── Shape & types ──────────────────────────────────────────
        meta["shape"] = {"rows": int(len(df)), "columns": int(len(df.columns))}
        meta["column_types"] = {
            k: v for k, v in col_types.items() if v
        }
        meta["missing_summary"] = {
            "total_missing_pct": round(
                df.isna().sum().sum() / max(df.size, 1) * 100, 2
            ),
            "duplicates_pct": round(
                df.duplicated().sum() / max(len(df), 1) * 100, 2
            ),
        }

        # ── Cardinality ────────────────────────────────────────────
        meta["cardinality"] = {
            col: int(df[col].nunique()) for col in df.columns
        }

        # ── Descriptive stats (numeric) ────────────────────────────
        if not desc_stats.empty:
            stat_keys = ["mean", "std", "min", "max", "50%",
                         "skewness", "kurtosis", "missing_%"]
            stat_keys = [k for k in stat_keys if k in desc_stats.columns]
            stats_dict = {}
            for col in desc_stats.index:
                row = {}
                for k in stat_keys:
                    try:
                        val = float(desc_stats.loc[col, k])
                        row[k] = None if (np.isnan(val) or np.isinf(val)) else round(val, 4)
                    except Exception:
                        row[k] = None
                stats_dict[col] = row
            meta["numeric_stats"] = stats_dict

        # ── Categorical top-values ─────────────────────────────────
        cat_info = {}
        for col in col_types.get("categorical", []):
            if col in df.columns:
                vc = df[col].value_counts().head(5)
                cat_info[col] = {
                    "top_values": vc.index.tolist(),
                    "top_counts": vc.values.tolist(),
                    "n_unique": int(df[col].nunique()),
                    "missing_%": round(df[col].isna().mean() * 100, 2),
                }
        if cat_info:
            meta["categorical_info"] = cat_info

        # ── Missing by column ──────────────────────────────────────
        if not missing_df.empty:
            meta["missing_by_column"] = (
                missing_df[["Column", "Missing %"]]
                .head(10)
                .to_dict(orient="records")
            )

        # ── Strong correlations ────────────────────────────────────
        if not strong_corrs.empty:
            meta["strong_correlations"] = (
                strong_corrs.head(10).to_dict(orient="records")
            )

        # ── Datetime ranges ────────────────────────────────────────
        dt_info = {}
        for col in col_types.get("datetime", []):
            if col in df.columns:
                try:
                    dt_series = pd.to_datetime(df[col], errors="coerce")
                    dt_info[col] = {
                        "min": str(dt_series.min()),
                        "max": str(dt_series.max()),
                        "missing_%": round(df[col].isna().mean() * 100, 2),
                    }
                except Exception:
                    pass
        if dt_info:
            meta["datetime_info"] = dt_info

        # ── Sample rows ────────────────────────────────────────────
        try:
            sample = df.head(3).to_dict(orient="records")
            meta["sample_rows"] = [
                {k: (None if (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else v)
                 for k, v in row.items()}
                for row in sample
            ]
        except Exception:
            pass

        return json.dumps(meta, cls=_SafeEncoder, ensure_ascii=False)

    def get_insights(self, metadata_json: str) -> Dict:
        """
        Send metadata to the Anthropic API and parse the response.

        Returns
        -------
        {"success": True,  "insights": {...}}   on success
        {"success": False, "error": "..."}      on failure
        """
        if not self.is_ready():
            return {
                "success": False,
                "error": (
                    "Anthropic client not initialised.  "
                    "Please provide a valid API key."
                ),
            }

        user_msg = (
            "Analyse this dataset metadata and return ONLY the JSON insights object:\n\n"
            + metadata_json
        )

        try:
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=2500,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                lines = raw.splitlines()
                raw   = "\n".join(
                    l for l in lines
                    if not l.strip().startswith("```")
                ).strip()

            insights = json.loads(raw)
            return {"success": True, "insights": insights}

        except json.JSONDecodeError as exc:
            return {"success": False, "error": f"JSON parse error: {exc}"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}
