"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 6 - Data Analysis with pandas.

This module offers a slightly more realistic collection of helpers
than the focused snippets in the chapter:

- loading CSV data with basic validation,
- computing returns and grouped summaries,
- and bridging between ``pandas`` and ``NumPy`` with explicit types.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd


def load_prices_csv(path: Path) -> pd.DataFrame:
    """Load a prices CSV with ``date`` index and basic sanity checks."""

    try:
        df = pd.read_csv(
            path,
            parse_dates=["date"],
            index_col="date",
        )
    except FileNotFoundError as exc:
        msg = f"input file not found: {exc.filename}"
        raise RuntimeError(msg) from exc
    except pd.errors.ParserError as exc:
        msg = f"CSV parsing failed for {path}: {exc}"
        raise RuntimeError(msg) from exc

    if "price" not in df.columns:
        raise ValueError("expected a 'price' column in the CSV data")

    return df.sort_index()


def compute_simple_returns(series: pd.Series) -> pd.Series:
    """Return a clean simple-returns ``Series`` from a price series."""

    rets = series.pct_change()
    if rets.notna().any():
        first_idx = rets.first_valid_index()
        if first_idx is not None:
            expected = (series.loc[first_idx] / series.iloc[0]) - 1.0
            assert np.isclose(rets.loc[first_idx], expected)
    return rets


def prices_to_array(series: pd.Series) -> npt.NDArray[np.floating]:
    """Convert a ``Series`` of prices to a floating-point array."""

    return series.to_numpy(dtype=float)


def summarize_by_symbol(df: pd.DataFrame) -> pd.DataFrame:
    """Compute basic per-symbol statistics for a long-format quotes table."""

    grouped = df.groupby("symbol")["price"].agg(["mean", "min", "max"])
    return grouped.sort_index()


def main() -> None:
    """Run tiny in-memory examples to sanity-check the helpers."""

    idx = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    prices = pd.Series([100.0, 101.5, 103.0], index=idx, name="price")

    rets = compute_simple_returns(prices)
    assert rets.iloc[1] > 0

    arr = prices_to_array(prices)
    assert arr.shape == (3,)

    quotes = pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-02", "2026-01-05"],
            "symbol": ["AAPL", "MSFT", "AAPL"],
            "price": [180.0, 350.0, 182.0],
        }
    )
    quotes["date"] = pd.to_datetime(quotes["date"])
    stats = summarize_by_symbol(quotes)
    assert "AAPL" in stats.index


if __name__ == "__main__":
    main()

