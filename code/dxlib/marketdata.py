"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 31 - Market-Based Valuation.

Utilities to load an options snapshot and extract a small implied-vol surface.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import datetime as dt
import math
import sys
from dataclasses import dataclass
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    package_dir = Path(__file__).resolve().parent
    sys.path = [
        path
        for path in sys.path
        if Path(path or ".").resolve() != package_dir
    ]
    sys.path.insert(0, str(package_dir.parent))
    __package__ = "dxlib"

import numpy as np
import pandas as pd

from .black_scholes import implied_vol_forward
from .time import time_to_maturity

__all__ = [
    "OptionsSnapshot",
    "load_spx_snapshot",
    "parity_table",
    "estimate_forward",
    "select_small_surface",
]


@dataclass(frozen=True, slots=True)
class OptionsSnapshot:
    """
    Cleaned options snapshot with metadata used throughout Chapter 31.
    """

    pricing_date: dt.date
    raw: pd.DataFrame


def load_spx_snapshot(path: str | Path) -> OptionsSnapshot:
    """
    Load and clean the SPX options snapshot CSV.
    """

    df = pd.read_csv(path)
    df = df.copy()
    df["PUTCALLIND"] = df["PUTCALLIND"].astype(str).str.strip().str.upper()
    df["EXPIR_DATE"] = pd.to_datetime(df["EXPIR_DATE"]).dt.date
    df["HSTCLSDATE"] = pd.to_datetime(df["HSTCLSDATE"]).dt.date
    df["BID"] = df["BID"].astype(float)
    df["ASK"] = df["ASK"].astype(float)
    df["MID"] = 0.5 * (df["BID"].astype(float) + df["ASK"].astype(float))
    df["SPREAD"] = df["ASK"] - df["BID"]
    df["HALF_SPREAD"] = 0.5 * df["SPREAD"]
    pricing_date = df["HSTCLSDATE"].iloc[0]
    return OptionsSnapshot(pricing_date=pricing_date, raw=df)


def parity_table(snapshot: OptionsSnapshot, expiry: dt.date) -> pd.DataFrame:
    """
    Merge calls and puts by strike for a single expiry.
    """

    df = snapshot.raw
    sub = df[df["EXPIR_DATE"] == expiry].copy()
    calls = sub[sub["PUTCALLIND"] == "CALL"].copy()
    puts = sub[sub["PUTCALLIND"] == "PUT"].copy()

    calls = calls.rename(
        columns={
            "MID": "CALL_MID",
            "BID": "CALL_BID",
            "ASK": "CALL_ASK",
            "SPREAD": "CALL_SPREAD",
            "HALF_SPREAD": "CALL_HALF_SPREAD",
            "DELTA": "CALL_DELTA",
            "MID_IV": "CALL_MID_IV",
        }
    )
    puts = puts.rename(
        columns={
            "MID": "PUT_MID",
            "BID": "PUT_BID",
            "ASK": "PUT_ASK",
            "SPREAD": "PUT_SPREAD",
            "HALF_SPREAD": "PUT_HALF_SPREAD",
            "DELTA": "PUT_DELTA",
            "MID_IV": "PUT_MID_IV",
        }
    )

    out = pd.merge(
        calls[
            [
                "EXPIR_DATE",
                "STRIKE_PRC",
                "CALL_BID",
                "CALL_ASK",
                "CALL_MID",
                "CALL_SPREAD",
                "CALL_HALF_SPREAD",
                "CALL_DELTA",
                "CALL_MID_IV",
            ]
        ],
        puts[
            [
                "EXPIR_DATE",
                "STRIKE_PRC",
                "PUT_BID",
                "PUT_ASK",
                "PUT_MID",
                "PUT_SPREAD",
                "PUT_HALF_SPREAD",
                "PUT_DELTA",
                "PUT_MID_IV",
            ]
        ],
        on=["EXPIR_DATE", "STRIKE_PRC"],
        how="inner",
    )
    out = out.rename(columns={"STRIKE_PRC": "STRIKE"})
    return out.sort_values("STRIKE").reset_index(drop=True)


def estimate_forward(
    parity: pd.DataFrame,
    *,
    discount_factor: float,
    min_price: float = 0.5,
    delta_band: tuple[float, float] = (0.2, 0.8),
) -> float:
    """
    Estimate the forward from put-call parity across a filtered strike band.
    """

    df = float(discount_factor)
    if df <= 0:
        raise ValueError("discount_factor must be positive")

    lo, hi = float(delta_band[0]), float(delta_band[1])
    work = parity.copy()
    work = work[(work["CALL_MID"] > min_price) & (work["PUT_MID"] > min_price)]
    work = work[(work["CALL_DELTA"] >= lo) & (work["CALL_DELTA"] <= hi)]
    if work.empty:
        raise ValueError("No suitable strikes to estimate the forward")

    fwd = work["STRIKE"] + (work["CALL_MID"] - work["PUT_MID"]) / df
    fwd = fwd.to_numpy(dtype=float)
    return float(np.median(fwd))


def select_small_surface(
    snapshot: OptionsSnapshot,
    expiries: list[dt.date],
    *,
    rate: float,
    moneyness_targets: list[float] | None = None,
    calibrate_to: str = "CALL",
    min_mid_price: float = 0.5,
    short_ttm_threshold: float | None = None,
    short_min_mid_price: float | None = None,
) -> pd.DataFrame:
    """
    Select a 3x5 quotes surface (calls + puts) for the book examples.
    """

    if moneyness_targets is None:
        moneyness_targets = [0.8, 0.9, 1.0, 1.1, 1.2]
    if min_mid_price <= 0:
        raise ValueError("min_mid_price must be positive")
    if short_ttm_threshold is not None and float(short_ttm_threshold) <= 0.0:
        raise ValueError("short_ttm_threshold must be positive")
    if short_min_mid_price is not None and float(short_min_mid_price) <= 0.0:
        raise ValueError("short_min_mid_price must be positive")

    def select_strikes(
        frame: pd.DataFrame,
        *,
        targets: list[float],
        mid_col: str,
        min_mid: float,
        n: int = 5,
    ) -> list[float]:
        eligible = frame[frame[mid_col] >= float(min_mid)].copy()
        if eligible.empty:
            raise ValueError("No eligible quotes after min_mid_price filter")

        strikes_out: list[float] = []
        for target in targets:
            work = eligible.copy()
            work["dist"] = (work["MNY"] - float(target)).abs()
            work = work.sort_values("dist")
            for strike in work["STRIKE"].to_numpy(dtype=float):
                if float(strike) not in strikes_out:
                    strikes_out.append(float(strike))
                    break
            if len(strikes_out) >= n:
                break

        if len(strikes_out) < n:
            remaining = eligible.copy()
            remaining = remaining[~remaining["STRIKE"].isin(strikes_out)]
            remaining = remaining.assign(
                abs_mny=(remaining["MNY"] - 1.0).abs(),
            )
            remaining = remaining.sort_values("abs_mny")
            for strike in remaining["STRIKE"].to_numpy(dtype=float):
                strikes_out.append(float(strike))
                if len(strikes_out) >= n:
                    break

        return sorted(strikes_out)[:n]

    results: list[pd.DataFrame] = []
    for expiry in expiries:
        parity = parity_table(snapshot, expiry)
        ttm = time_to_maturity(snapshot.pricing_date, expiry)
        expiry_min_mid = float(min_mid_price)
        if (
            short_ttm_threshold is not None
            and short_min_mid_price is not None
            and float(ttm) <= float(short_ttm_threshold)
        ):
            expiry_min_mid = max(expiry_min_mid, float(short_min_mid_price))
        df = math.exp(-float(rate) * float(ttm))
        fwd = estimate_forward(
            parity,
            discount_factor=df,
            min_price=expiry_min_mid,
        )

        parity = parity.copy()
        parity["TTM"] = float(ttm)
        parity["DF"] = float(df)
        parity["FWD"] = float(fwd)
        parity["MNY"] = parity["STRIKE"] / float(fwd)

        opt = str(calibrate_to).strip().upper()
        if opt not in {"CALL", "PUT"}:
            raise ValueError("calibrate_to must be 'CALL' or 'PUT'")

        base_targets = list(moneyness_targets)
        eligible = parity[parity[f"{opt}_MID"] >= float(expiry_min_mid)]
        max_mny = float(eligible["MNY"].max())
        if max_mny < 1.15:
            base_targets = [0.8, 0.9, 1.0, 1.05, 1.1]

        strikes = select_strikes(
            parity,
            targets=base_targets,
            mid_col=f"{opt}_MID",
            min_mid=expiry_min_mid,
            n=5,
        )
        small = parity[parity["STRIKE"].isin(strikes)].copy()

        ivs: list[float] = []
        for _, row in small.iterrows():
            price = float(row[f"{opt}_MID"])
            iv = implied_vol_forward(
                price=price,
                forward=float(row["FWD"]),
                strike=float(row["STRIKE"]),
                maturity=float(row["TTM"]),
                option_type="call" if opt == "CALL" else "put",
                discount_factor=float(row["DF"]),
            )
            ivs.append(iv)
        small["CALIBRATE_TO"] = opt
        small["IMPL_VOL"] = np.array(ivs, dtype=float)
        results.append(small)

    out = pd.concat(results, axis=0, ignore_index=True)
    return out
