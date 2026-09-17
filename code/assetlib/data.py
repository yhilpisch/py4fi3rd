"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 21 - A Small Asset Management Library in Python.

Market data loading and return calculation utilities.

The default data path resolves relative to the project layout
(`PROJECT_ROOT / data / eod_data.csv`), so the package expects to live
under `code/assetlib/` in the book's repository; pass an explicit path
to `MarketData.load()` to read prices from anywhere else.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # project root
DATA_PATH = PROJECT_ROOT / "data" / "eod_data.csv"  # default EOD CSV path


@dataclass
class MarketData:
    """Helper for loading and working with end-of-day prices."""

    prices: pd.DataFrame

    @classmethod
    def load(cls, path: Optional[str] = None) -> "MarketData":
        """Load end-of-day prices from a CSV file."""

        csv_path = Path(path) if path is not None else DATA_PATH  # file path
        df = pd.read_csv(
            csv_path, parse_dates=["Date"], index_col="Date"
        )  # CSV
        df.sort_index(inplace=True)  # ensure time order
        return cls(prices=df)  # type: ignore[arg-type]

    def select(
        self,
        symbols: Sequence[str],
        dropna: bool = True,
    ) -> pd.DataFrame:
        """Select symbol columns, optionally dropping incomplete rows."""

        cols: List[str] = list(symbols)  # concrete list of symbols
        sub = self.prices[cols]  # price slice
        if dropna:
            sub = sub.dropna(how="any")
        return sub

    def returns(
        self,
        symbols: Sequence[str],
        dropna: bool = True,
    ) -> pd.DataFrame:
        """Compute daily simple returns for the selected symbols."""

        prices = self.select(symbols=symbols, dropna=dropna)  # aligned prices
        rets = prices.pct_change()  # simple returns
        if dropna:
            rets = rets.dropna(how="any")
        return rets

    def window(self, n_days: int) -> "MarketData":
        if n_days <= 0:
            raise ValueError("n_days must be positive")
        window_df = self.prices.iloc[-n_days:].copy()  # trailing window
        return MarketData(prices=window_df)  # new MarketData instance

    def to_long(self, symbols: Optional[Iterable[str]] = None) -> pd.DataFrame:
        if symbols is not None:
            df = self.prices[list(symbols)]  # restricted universe
        else:
            df = self.prices  # all columns
        stacked = df.stack().to_frame("price")  # wide to long
        stacked.index.names = ["Date", "symbol"]  # explicit index names
        return stacked
