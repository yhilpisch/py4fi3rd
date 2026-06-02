"""Python for Finance, 3rd ed., O'Reilly (2026).
Lab 02 - Summarizing Earnings Surprises with LLMs and Python.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TEXT_FILE = ROOT / "data" / "earnings_text_samples.csv"
SUMMARY_FILE = ROOT / "data" / "earnings_summary_samples.json"


def load_samples() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads the raw text sample and the structured summary sample."""
    text_df = pd.read_csv(TEXT_FILE)
    summary_df = pd.read_json(SUMMARY_FILE)
    return text_df, summary_df


def normalize_risk_flags(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Converts list-valued risk flags into a compact display string."""
    out = summary_df.copy()
    out["risk_flags"] = out["risk_flags"].apply(lambda xs: ", ".join(xs))
    return out


def build_joined_view() -> pd.DataFrame:
    """Joins raw text items with structured summaries for review."""
    text_df, summary_df = load_samples()
    joined = text_df.merge(summary_df, on=["ticker", "report_date"], how="left")
    return normalize_risk_flags(joined)


def validate_summary_fields(summary_df: pd.DataFrame) -> dict[str, object]:
    """Performs lightweight schema checks for the sample summary file."""
    required = {
        "ticker",
        "report_date",
        "summary",
        "surprise_direction",
        "guidance_change",
        "main_positive_driver",
        "main_negative_driver",
        "risk_flags",
        "confidence_note",
    }
    missing_cols = sorted(required - set(summary_df.columns))
    null_counts = summary_df[list(required)].isna().sum().to_dict()
    return {
        "rows": int(len(summary_df)),
        "missing_columns": missing_cols,
        "null_counts": null_counts,
    }


def main() -> None:
    text_df, summary_df = load_samples()
    joined = build_joined_view()
    checks = validate_summary_fields(summary_df)

    print("RAW SAMPLE")
    print(text_df[["ticker", "report_date", "source"]].to_string(index=False))
    print()

    print("SUMMARY CHECKS")
    print(json.dumps(checks, indent=2))
    print()

    print("JOINED VIEW")
    cols = [
        "ticker",
        "surprise_direction",
        "guidance_change",
        "main_positive_driver",
        "main_negative_driver",
        "risk_flags",
    ]
    print(joined[cols].to_string(index=False))


if __name__ == "__main__":
    main()
