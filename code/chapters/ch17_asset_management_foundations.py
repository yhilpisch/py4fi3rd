"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 17 - Asset Management Foundations.

Companion code for core examples in Chapter 17:

- tracking error for a simple equally weighted portfolio vs `SPY`
- cross-sectional holdings snapshot and sector weights
- long-format prices example via `stack()`
- minimal universe metadata table

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_EOD = ROOT / "data" / "eod_data.csv"


def load_eod_data() -> pd.DataFrame:
    """Load end-of-day prices for the book dataset."""

    if not DATA_EOD.exists():
        msg = f"Expected EOD data at {DATA_EOD}; file not found."
        raise FileNotFoundError(msg)
    prices = pd.read_csv(DATA_EOD, parse_dates=["Date"], index_col="Date")
    return prices.dropna(how="any")


def tracking_error_example() -> float:
    """Compute annualised tracking error for the Chapter 17 example."""

    prices = load_eod_data()
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
    return te_annual


def holdings_snapshot() -> pd.DataFrame:
    """Return the small holdings snapshot used in the chapter."""

    holdings = pd.DataFrame(
        {
            "symbol": ["AAPL", "NVDA", "JPM", "SPY"],
            "quantity": [120, 80, 150, 200],
            "price": [180.25, 820.10, 145.30, 520.10],
            "sector": [
                "Technology",
                "Technology",
                "Financials",
                "Equity Index",
            ],
            "region": ["US", "US", "US", "Global"],
            "currency": ["USD", "USD", "USD", "USD"],
        }
    )
    holdings["market_value"] = holdings["quantity"] * holdings["price"]
    holdings["weight"] = (
        holdings["market_value"] / holdings["market_value"].sum()
    )
    return holdings


def sector_weights_from_holdings(holdings: pd.DataFrame) -> pd.Series:
    """Aggregate sector weights from a holdings snapshot."""

    return holdings.groupby("sector")["weight"].sum()


def stacked_prices_example() -> pd.DataFrame:
    """Return the head of the stacked prices panel."""

    prices = load_eod_data()
    prices = prices[["AAPL", "NVDA", "JPM", "SPY"]]
    stacked = prices.stack().to_frame("price")
    stacked.index.names = ["Date", "symbol"]
    return stacked.head()


def universe_table() -> pd.DataFrame:
    """Return the small universe metadata table used in the chapter."""

    universe = pd.DataFrame(
        {
            "instrument_id": [1, 2, 3, 4],
            "symbol": ["AAPL", "NVDA", "JPM", "SPY"],
            "asset_class": ["Equity", "Equity", "Equity", "Equity ETF"],
            "region": ["US", "US", "US", "Global"],
            "esg_flag": [False, False, False, False],
        }
    ).set_index("instrument_id")
    return universe


def main() -> None:
    """Run a compact Chapter 17 demo and print key results."""

    print("== Tracking error example ==")
    te_annual = tracking_error_example()
    print(f"Annualised tracking error: {te_annual:.6f} ({te_annual:.4%})")
    print()

    print("== Holdings snapshot ==")
    holdings = holdings_snapshot()
    print(holdings)
    print()

    print("== Sector weights ==")
    sectors = sector_weights_from_holdings(holdings)
    print(sectors)
    print()

    print("== Stacked prices head ==")
    stacked = stacked_prices_example()
    print(stacked)
    print()

    print("== Universe table ==")
    universe = universe_table()
    print(universe)

    assert 0 < te_annual < 1, "tracking error must be between 0 and 1"
    assert abs(float(holdings["weight"].sum()) - 1.0) < 1e-6, (
        "weights must sum to 1"
    )


if __name__ == "__main__":
    main()
