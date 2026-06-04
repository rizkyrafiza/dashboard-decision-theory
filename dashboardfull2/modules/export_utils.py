"""
modules/export_utils.py
─────────────────────────────────────────────
Download helpers: multi-sheet Excel report
and per-table CSV export.
"""

import io
import json
from typing import Dict, List, Optional

import pandas as pd


class ExportUtils:
    """Generates in-memory download bytes for various export formats."""

    # ─────────────────────────────────────────────────────────────
    # EXCEL REPORT (multi-sheet)
    # ─────────────────────────────────────────────────────────────

    def to_excel_report(
        self,
        df: pd.DataFrame,
        desc_stats: pd.DataFrame,
        missing_df: pd.DataFrame,
        outlier_df: pd.DataFrame,
        corr_pearson: pd.DataFrame,
        cat_analysis: Dict[str, pd.DataFrame],
        col_types: Dict[str, List[str]],
        ai_insights: Optional[Dict] = None,
    ) -> bytes:
        """
        Build a multi-sheet Excel workbook and return raw bytes.
        Sheets:
          1. Dataset Sample (first 1 000 rows)
          2. Column Types
          3. Descriptive Statistics
          4. Missing Values
          5. Outlier Detection
          6. Correlation (Pearson)
          7-N. Categorical Value Counts
          N+1. AI Insights (if available)
        """
        buf = io.BytesIO()

        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            wb  = writer.book
            hdr = wb.add_format({
                "bold": True, "bg_color": "#0f172a", "font_color": "#00d4ff",
                "border": 1,
            })
            cell = wb.add_format({"border": 1, "font_color": "#1e293b"})

            def _write(df_: pd.DataFrame, sheet: str, col_w: int = 22):
                if df_.empty:
                    pd.DataFrame({"Note": ["No data available"]}).to_excel(
                        writer, sheet_name=sheet, index=False
                    )
                    return
                df_.to_excel(writer, sheet_name=sheet, index=True)
                ws = writer.sheets[sheet]
                ws.set_column(0, len(df_.columns), col_w)

            # 1. Sample
            sample = df.head(1000).copy()
            sample.to_excel(writer, sheet_name="Dataset Sample", index=False)
            ws0 = writer.sheets["Dataset Sample"]
            ws0.set_column(0, len(df.columns), 18)

            # 2. Column Types
            type_rows = []
            for dtype, cols in col_types.items():
                for c in cols:
                    type_rows.append({"Column": c, "Detected Type": dtype})
            _write(pd.DataFrame(type_rows), "Column Types", col_w=26)

            # 3. Descriptive Stats
            _write(desc_stats.reset_index().rename(columns={"index": "Column"}),
                   "Descriptive Statistics", col_w=20)

            # 4. Missing Values
            _write(missing_df, "Missing Values", col_w=22)

            # 5. Outliers
            _write(outlier_df, "Outlier Detection", col_w=22)

            # 6. Correlation
            _write(corr_pearson, "Correlation (Pearson)", col_w=20)

            # 7-N. Categorical
            for col, tbl in list(cat_analysis.items())[:8]:
                safe = col[:28]  # Excel sheet name ≤ 31 chars
                _write(tbl, f"Cat_{safe}", col_w=20)

            # AI Insights sheet
            if ai_insights and ai_insights.get("success"):
                ins = ai_insights["insights"]
                rows = []

                # Domain
                dom = ins.get("domain", {})
                rows += [
                    ["=== DOMAIN ===", ""],
                    ["Detected Domain", dom.get("detected_domain", "")],
                    ["Confidence",      dom.get("confidence", "")],
                    ["Reasoning",       dom.get("reasoning", "")],
                    ["", ""],
                ]

                # Quality
                qual = ins.get("dataset_quality", {})
                rows += [
                    ["=== DATA QUALITY ===", ""],
                    ["Score",  f"{qual.get('score', '')}/10"],
                    ["Label",  qual.get("label", "")],
                    ["", ""],
                ]
                for s in qual.get("strengths", []):
                    rows.append(["Strength", s])
                for c in qual.get("concerns", []):
                    rows.append(["Concern", c])
                rows.append(["", ""])

                # Key Insights
                rows.append(["=== KEY INSIGHTS ===", ""])
                for ki in ins.get("key_insights", []):
                    rows += [
                        ["Title",    ki.get("title", "")],
                        ["Type",     ki.get("type", "")],
                        ["Importance", ki.get("importance", "")],
                        ["Description", ki.get("description", "")],
                        ["Columns",  ", ".join(ki.get("affected_columns", []))],
                        ["", ""],
                    ]

                # Business Questions
                rows.append(["=== BUSINESS QUESTIONS ===", ""])
                for q in ins.get("business_questions", []):
                    rows.append(["Question", q])
                rows.append(["", ""])

                # Next Steps
                rows.append(["=== NEXT STEPS ===", ""])
                for ns in ins.get("next_steps", []):
                    rows.append(["Step", ns])

                ai_df = pd.DataFrame(rows, columns=["Category", "Detail"])
                ai_df.to_excel(writer, sheet_name="AI Insights", index=False)
                ws_ai = writer.sheets["AI Insights"]
                ws_ai.set_column(0, 0, 28)
                ws_ai.set_column(1, 1, 80)

        buf.seek(0)
        return buf.read()

    # ─────────────────────────────────────────────────────────────
    # CSV EXPORT (single DataFrame)
    # ─────────────────────────────────────────────────────────────

    def to_csv(self, df: pd.DataFrame) -> bytes:
        return df.to_csv(index=True).encode("utf-8-sig")

    # ─────────────────────────────────────────────────────────────
    # AI INSIGHTS JSON
    # ─────────────────────────────────────────────────────────────

    def insights_to_json(self, ai_insights: Dict) -> bytes:
        return json.dumps(ai_insights, indent=2, ensure_ascii=False).encode("utf-8")