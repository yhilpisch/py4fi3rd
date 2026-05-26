"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 17 - Asset Management Foundations.

Compute tracking error for an equally weighted portfolio vs SPY
over a recent two-year window of daily data.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd


def main() -> None:
    """Compute and print annualised tracking error for a simple example."""

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    local = base_dir / "data" / "eod_data.csv"
    remote = "https://hilpisch.com/eod_data.csv"
    source: str | pathlib.Path = local if local.exists() else remote

    prices = pd.read_csv(
        source,
        parse_dates=["Date"],
        index_col="Date",
    ).dropna(how="any")

    symbols = ["AAPL", "JPM", "TLT"]
    sub = prices[symbols + ["SPY"]]

    rets_all = sub.pct_change().dropna()

    max_rows = 2 * 252
    if len(rets_all) > max_rows:
        rets = rets_all.iloc[-max_rows:]
    else:
        rets = rets_all

    w_eq = np.repeat(1.0 / len(symbols), len(symbols))
    r_port = (rets[symbols] * w_eq).sum(axis=1)
    r_bench = rets["SPY"]
    active = r_port - r_bench

    te_annual = float(active.std(ddof=1) * np.sqrt(252.0))

    print(f"Annualised tracking error: {te_annual:.6f} ({te_annual:.2%})")
    assert te_annual > 0, "tracking error must be positive"
    assert np.isfinite(te_annual), "tracking error must be finite"


if __name__ == "__main__":
    main()
