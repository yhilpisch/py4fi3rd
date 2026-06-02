"""Python for Finance, 3rd ed., O'Reilly (2026).
Lab 07 - Constructing the VIX from SPX Option Data.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "spx_options_snapshot.csv"
RATE = 0.04
TARGET_DAYS = 30
YEAR_DAYS = 365


@dataclass(frozen=True, slots=True)
class VixTerm:
    """Container for one maturity's model-free variance calculation."""

    expiry: pd.Timestamp
    days: int
    ttm: float
    forward: float
    k0: float
    variance: float
    contribution_table: pd.DataFrame


def load_option_snapshot() -> tuple[pd.Timestamp, pd.DataFrame]:
    """Load and clean the local SPX option snapshot."""
    raw = pd.read_csv(DATA_FILE)
    df = raw.copy()
    df["PUTCALLIND"] = df["PUTCALLIND"].astype(str).str.strip().str.upper()
    df["EXPIR_DATE"] = pd.to_datetime(df["EXPIR_DATE"])
    df["HSTCLSDATE"] = pd.to_datetime(df["HSTCLSDATE"])
    df["BID"] = pd.to_numeric(df["BID"], errors="coerce")
    df["ASK"] = pd.to_numeric(df["ASK"], errors="coerce")
    df["MID"] = 0.5 * (df["BID"] + df["ASK"])
    pricing_date = df["HSTCLSDATE"].dropna().iloc[0]
    return pricing_date, df


def available_expiries(df: pd.DataFrame) -> pd.DataFrame:
    """Return available expiries with calendar days from the snapshot date."""
    pricing_date = df["HSTCLSDATE"].dropna().iloc[0]
    expiries = pd.Series(sorted(df["EXPIR_DATE"].dropna().unique()))
    out = pd.DataFrame({"expiry": pd.to_datetime(expiries)})
    out["days"] = (out["expiry"] - pricing_date).dt.days
    return out[out["days"] > 0].reset_index(drop=True)


def select_vix_expiries(
    df: pd.DataFrame,
    target_days: int = TARGET_DAYS,
) -> pd.DataFrame:
    """Select the listed expiries immediately around the target horizon."""
    expiries = available_expiries(df)
    lower = expiries[expiries["days"] <= target_days].tail(1)
    upper = expiries[expiries["days"] >= target_days].head(1)
    selected = pd.concat([lower, upper]).drop_duplicates("expiry")
    if len(selected) < 2:
        raise ValueError("Need two expiries around the target horizon")
    return selected.reset_index(drop=True)


def parity_table(df: pd.DataFrame, expiry: pd.Timestamp) -> pd.DataFrame:
    """Merge calls and puts for one expiry by strike."""
    sub = df[df["EXPIR_DATE"] == expiry].copy()
    calls = sub[sub["PUTCALLIND"] == "CALL"].copy()
    puts = sub[sub["PUTCALLIND"] == "PUT"].copy()
    calls = calls.rename(
        columns={"BID": "CALL_BID", "ASK": "CALL_ASK", "MID": "CALL_MID"}
    )
    puts = puts.rename(
        columns={"BID": "PUT_BID", "ASK": "PUT_ASK", "MID": "PUT_MID"}
    )
    cols = ["STRIKE_PRC", "CALL_BID", "CALL_ASK", "CALL_MID"]
    pcols = ["STRIKE_PRC", "PUT_BID", "PUT_ASK", "PUT_MID"]
    merged = pd.merge(calls[cols], puts[pcols], on="STRIKE_PRC", how="inner")
    merged = merged.rename(columns={"STRIKE_PRC": "strike"})
    merged = merged.dropna(subset=["CALL_MID", "PUT_MID"])
    return merged.sort_values("strike").reset_index(drop=True)


def estimate_forward(parity: pd.DataFrame, ttm: float, rate: float) -> float:
    """Estimate the forward from the strike with the smallest call-put gap."""
    work = parity.copy()
    work["abs_gap"] = (work["CALL_MID"] - work["PUT_MID"]).abs()
    row = work.sort_values(["abs_gap", "strike"]).iloc[0]
    discount = np.exp(-rate * ttm)
    return float(row["strike"] + (row["CALL_MID"] - row["PUT_MID"]) / discount)


def compute_delta_k(strikes: pd.Series) -> np.ndarray:
    """Compute VIX strike intervals for a sorted strike grid."""
    values = strikes.to_numpy(dtype=float)
    delta = np.empty_like(values)
    delta[0] = values[1] - values[0]
    delta[-1] = values[-1] - values[-2]
    delta[1:-1] = 0.5 * (values[2:] - values[:-2])
    return delta


def select_otm_strip(parity: pd.DataFrame, forward: float) -> pd.DataFrame:
    """Build the OTM option strip used in the variance sum."""
    k0 = float(parity.loc[parity["strike"] <= forward, "strike"].max())
    work = parity.copy()
    lower_rows = []
    zero_bids = 0
    lower = work[work["strike"] < k0].sort_values("strike", ascending=False)
    for row in lower.itertuples(index=False):
        if row.PUT_BID <= 0.0:
            zero_bids += 1
            if zero_bids >= 2:
                break
        else:
            zero_bids = 0
            lower_rows.append(
                {
                    "strike": row.strike,
                    "option_bid": row.PUT_BID,
                    "option_mid": row.PUT_MID,
                }
            )

    upper_rows = []
    zero_bids = 0
    upper = work[work["strike"] > k0].sort_values("strike")
    for row in upper.itertuples(index=False):
        if row.CALL_BID <= 0.0:
            zero_bids += 1
            if zero_bids >= 2:
                break
        else:
            zero_bids = 0
            upper_rows.append(
                {
                    "strike": row.strike,
                    "option_bid": row.CALL_BID,
                    "option_mid": row.CALL_MID,
                }
            )

    center = work[work["strike"] == k0].iloc[0]
    rows = list(reversed(lower_rows))
    rows.append(
        {
            "strike": float(center["strike"]),
            "option_bid": 0.5 * (center["CALL_BID"] + center["PUT_BID"]),
            "option_mid": 0.5 * (center["CALL_MID"] + center["PUT_MID"]),
        }
    )
    rows.extend(upper_rows)
    strip = pd.DataFrame(rows)
    strip = strip.dropna(subset=["option_mid"])
    strip = strip[strip["option_mid"] > 0.0]
    return strip.sort_values("strike").reset_index(drop=True)


def compute_term_variance(
    df: pd.DataFrame,
    pricing_date: pd.Timestamp,
    expiry: pd.Timestamp,
    rate: float = RATE,
) -> VixTerm:
    """Compute model-free variance for one listed option maturity."""
    days = int((expiry - pricing_date).days)
    ttm = days / YEAR_DAYS
    parity = parity_table(df, expiry)
    forward = estimate_forward(parity, ttm=ttm, rate=rate)
    strip = select_otm_strip(parity, forward=forward)
    k0 = float(parity.loc[parity["strike"] <= forward, "strike"].max())

    table = strip.copy()
    table["delta_k"] = compute_delta_k(table["strike"])
    table["contribution"] = (
        table["delta_k"]
        / table["strike"] ** 2
        * np.exp(rate * ttm)
        * table["option_mid"]
    )
    sum_term = float(table["contribution"].sum())
    adjustment = (forward / k0 - 1.0) ** 2
    variance = (2.0 / ttm) * sum_term - (1.0 / ttm) * adjustment

    return VixTerm(
        expiry=expiry,
        days=days,
        ttm=ttm,
        forward=forward,
        k0=k0,
        variance=float(variance),
        contribution_table=table,
    )


def interpolate_30_day_variance(
    near: VixTerm,
    next_term: VixTerm,
    target_days: int = TARGET_DAYS,
) -> float:
    """Interpolate linearly in total variance to the target horizon."""
    n1 = near.days
    n2 = next_term.days
    nt = target_days
    total_var = (
        near.ttm
        * near.variance
        * (n2 - nt)
        / (n2 - n1)
        + next_term.ttm
        * next_term.variance
        * (nt - n1)
        / (n2 - n1)
    )
    return float(total_var * YEAR_DAYS / target_days)


def build_vix_calculation(rate: float = RATE) -> dict[str, object]:
    """Run the full two-maturity VIX construction workflow."""
    pricing_date, df = load_option_snapshot()
    expiries = select_vix_expiries(df)
    terms = [
        compute_term_variance(df, pricing_date, row.expiry, rate=rate)
        for row in expiries.itertuples(index=False)
    ]
    near, next_term = terms
    variance_30 = interpolate_30_day_variance(near, next_term)
    vix = 100.0 * np.sqrt(variance_30)
    return {
        "pricing_date": pricing_date,
        "terms": terms,
        "variance_30": variance_30,
        "vix": float(vix),
    }


def term_summary(calculation: dict[str, object]) -> pd.DataFrame:
    """Summarize the two maturity-specific variance calculations."""
    rows = []
    for term in calculation["terms"]:
        rows.append(
            {
                "expiry": term.expiry.date().isoformat(),
                "days": term.days,
                "forward": term.forward,
                "k0": term.k0,
                "variance": term.variance,
                "volatility": np.sqrt(term.variance),
                "option_count": len(term.contribution_table),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Print a compact summary of the VIX construction."""
    calc = build_vix_calculation()
    print(f"Pricing date: {calc['pricing_date'].date().isoformat()}")
    print()
    print(term_summary(calc).round(4).to_string(index=False))
    print()
    print(f"30-day variance: {calc['variance_30']:.6f}")
    print(f"VIX-style index level: {calc['vix']:.2f}")


if __name__ == "__main__":
    main()
