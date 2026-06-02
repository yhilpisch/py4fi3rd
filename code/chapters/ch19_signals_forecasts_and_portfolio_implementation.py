"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 19 - Signals, Forecasts, and Portfolio Implementation.

Companion code for the core examples in Chapter 19:

- feature engineering for momentum and volatility
- forward-return targets and information coefficients
- mapping standardised signals to weights
- simple weekly rebalancing backtest and turnover diagnostics

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_EOD = ROOT / "data" / "eod_data.csv"


def load_prices() -> pd.DataFrame:
    """Load end-of-day prices from the local CSV or remote fallback."""

    if not DATA_EOD.exists():
        remote = "https://hilpisch.com/eod_data.csv"
        prices = pd.read_csv(remote, parse_dates=["Date"], index_col="Date")
    else:
        prices = pd.read_csv(DATA_EOD, parse_dates=["Date"], index_col="Date")
    return prices


def prepare_universe() -> Tuple[pd.DataFrame, list[str]]:
    """Return returns and universe list for the small example."""

    prices = load_prices()
    universe = ["AAPL", "JPM", "TLT"]
    cols = universe + ["SPY"]
    sub = prices[cols].dropna(how="any")

    rets = sub[universe].pct_change().dropna()
    max_rows = 2 * 252
    if len(rets) > max_rows:
        rets = rets.iloc[-max_rows:]
    return rets, universe


def features_and_targets(
    rets: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute momentum features and forward-return targets."""

    mom_20d = rets.rolling(20).mean()
    fwd_5d = rets.shift(-5).rolling(5).sum()
    aligned_index = mom_20d.index.intersection(fwd_5d.index)
    mom_20d = mom_20d.loc[aligned_index]
    fwd_5d = fwd_5d.loc[aligned_index]
    return mom_20d, fwd_5d


def daily_ic(signal: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    """Compute a time series of daily Spearman rank information coefficients."""

    rows = []
    for date, x in signal.iterrows():
        if date not in target.index:
            continue
        aligned_xy = pd.concat([x, target.loc[date]], axis=1).dropna()
        if len(aligned_xy) < 2:
            continue
        rows.append(
            aligned_xy.iloc[:, 0].corr(
                aligned_xy.iloc[:, 1],
                method="spearman",
            )
        )
    return pd.Series(rows, name="ic")


def weights_from_row(row: pd.Series) -> np.ndarray:
    """Map a row of signals to long-only weights via z-scores."""

    z = (row - row.mean()) / row.std()
    z_pos = z.clip(lower=0)
    if z_pos.sum() == 0:
        return np.repeat(1.0 / len(row), len(row))
    return (z_pos / z_pos.sum()).values


def weekly_weights(
    mom_20d: pd.DataFrame,
    universe: list[str],
) -> pd.DataFrame:
    """Compute weekly signal-based weights."""

    weekly = mom_20d.resample("W-FRI").last().dropna()
    weights = weekly.apply(
        weights_from_row,
        axis=1,
        result_type="expand",
    )
    weights.columns = universe
    return weights


def turnover_series(weights: pd.DataFrame) -> pd.Series:
    """Return a series of daily turnover values."""

    diff = weights.diff().abs().sum(axis=1)
    return 0.5 * diff


def main() -> None:
    """Run a compact Chapter 19 demo and print key results."""

    rets, universe = prepare_universe()
    mom_20d, fwd_5d = features_and_targets(rets)

    print("== Feature tail (mom_20d) ==")
    print(mom_20d.tail())
    print()

    print("== Forward 5-day returns tail ==")
    print(fwd_5d.tail())
    print()

    ic = daily_ic(mom_20d, fwd_5d)
    print("== Information coefficient summary ==")
    desc = ic.describe().round(3)
    print(desc)
    print()

    latest = mom_20d.iloc[-1]
    z_scores = (latest - latest.mean()) / latest.std()
    z_pos = z_scores.clip(lower=0)
    if z_pos.sum() == 0:
        weights = pd.Series(
            np.repeat(1.0 / len(universe), len(universe)),
            index=universe,
        )
    else:
        weights = (z_pos / z_pos.sum()).reindex(universe)
    print("== Latest momentum z-scores ==")
    print(z_scores.reindex(universe))
    print()

    print("== Long-only weights from z-scores ==")
    print(weights)
    print()

    w_weekly = weekly_weights(mom_20d, universe)
    w_daily = w_weekly.reindex(rets.index, method="ffill")

    to_daily = turnover_series(w_daily)
    to_annual = float(to_daily.mean() * 252.0)
    print("== Annualised turnover for weekly rebalancing ==")
    print(f"{to_annual:.6f}")
    print()

    assert abs(float(weights.sum()) - 1.0) < 1e-6, "weights must sum to 1"
    assert to_annual > 0, "annualised turnover must be positive"


if __name__ == "__main__":
    main()
