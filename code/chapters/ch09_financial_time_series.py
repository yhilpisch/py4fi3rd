"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 9 - Financial Time Series.

This companion module collects reusable helpers for core time-series tasks:

- loading and cleaning daily price data,
- computing returns and rolling indicators,
- resampling and realised-volatility calculations,
- and evaluating a simple SMA timing rule.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_COLUMNS = ("AAPL", "NVDA", "SPY", "GLD", "TLT")


def synthetic_eod() -> pd.DataFrame:
    """Return a compact fallback price table when no local CSV is available."""

    idx = pd.date_range("2027-01-02", periods=8, freq="B", name="Date")
    return pd.DataFrame(
        {
            "AAPL": [190.0, 191.5, 193.0, 192.2, 194.1, 195.0, 196.3, 197.0],
            "NVDA": [620.0, 625.0, 632.0, 629.0, 638.0, 644.0, 649.0, 655.0],
            "SPY": [540.0, 541.0, 542.2, 541.8, 543.0, 544.1, 545.0, 546.2],
            "GLD": [210.0, 209.8, 210.5, 211.2, 210.9, 211.7, 212.0, 212.4],
            "TLT": [98.0, 98.4, 98.1, 98.7, 99.0, 98.8, 99.2, 99.5],
        },
        index=idx,
    )


def load_eod(path: str | Path | None = None) -> pd.DataFrame:
    """Load end-of-day prices from disk, with a small synthetic fallback."""

    if path is None:
        path = Path(__file__).resolve().parents[2] / "data" / "eod_data.csv"
    path = Path(path)
    if path.exists():
        df = pd.read_csv(
            path,
            parse_dates=["Date"],
            index_col="Date",
        ).dropna(how="all")
        df.index.name = "Date"
        return df
    return synthetic_eod()


def select_prices(
    df: pd.DataFrame,
    symbols: tuple[str, ...] = DATA_COLUMNS,
) -> pd.DataFrame:
    """Return a cleaned subset of the requested price columns."""

    cols = [sym for sym in symbols if sym in df.columns]
    return df[cols].dropna()


def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Compute log returns for a price object."""

    return np.log(prices / prices.shift(1)).dropna()


def rolling_indicators(
    series: pd.Series,
    short: int = 21,
    long: int = 63,
) -> pd.DataFrame:
    """Build a small indicator table with short/long SMAs."""

    out = pd.DataFrame({"price": series})
    out["sma_short"] = series.rolling(short).mean()
    out["sma_long"] = series.rolling(long).mean()
    return out.dropna()


def realized_volatility(
    rets: pd.Series,
    window: int = 21,
    annualization: int = 252,
) -> pd.Series:
    """Annualise rolling return volatility over the chosen window."""

    return rets.rolling(window).std() * np.sqrt(annualization)


def resample_last(prices: pd.Series, freq: str = "W") -> pd.Series:
    """Resample a price series by taking the last observation in each period."""

    return prices.resample(freq).last().dropna()


def sma_strategy_equity(
    series: pd.Series,
    short: int = 21,
    long: int = 63,
) -> pd.DataFrame:
    """Evaluate a long-or-flat SMA timing rule against buy-and-hold."""

    indicators = rolling_indicators(series, short=short, long=long)
    rets = series.pct_change().reindex(indicators.index).fillna(0.0)
    position = pd.Series(
        np.where(indicators["sma_short"] > indicators["sma_long"], 1.0, 0.0),
        index=indicators.index,
    )
    strat_rets = position.shift(1).fillna(0.0) * rets
    equity = pd.DataFrame(index=indicators.index)
    equity["strategy"] = (1.0 + strat_rets).cumprod()
    equity["buy_and_hold"] = (1.0 + rets).cumprod()
    return equity


def main() -> None:
    """Run a compact end-to-end demo of the chapter helpers."""

    prices = select_prices(load_eod())
    rets = log_returns(prices)

    spy = prices["SPY"] if "SPY" in prices.columns else prices.iloc[:, 0]
    spy_log = log_returns(spy)
    weekly = resample_last(spy)
    vol = realized_volatility(spy_log)
    equity = sma_strategy_equity(spy)

    print(prices.tail())
    print(rets.tail())
    print(weekly.tail())
    print(vol.dropna().tail())
    print(equity.tail())

    assert len(prices) > 0, "prices must not be empty"
    assert len(rets) < len(prices), "returns must be shorter than prices"
    assert (equity > 0).all().all(), "equity curves must be positive"


if __name__ == "__main__":
    main()
