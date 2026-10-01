"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 21 - A Small Asset Management Library in Python.

Simple rebalancing and backtesting utilities.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from assetlib.data import MarketData
else:
    from .data import MarketData


@dataclass
class BacktestResult:
    dates: pd.DatetimeIndex
    portfolio_returns: pd.Series
    benchmark_returns: pd.Series

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "portfolio": self.portfolio_returns,
                "benchmark": self.benchmark_returns,
            }
        )


class BacktestEngine:
    """Turn target weights into a simple buy-and-hold-with-rebalancing backtest.

    This implementation focuses on clarity and didactic value rather than
    handling all real-world trading details.
    """

    def __init__(
        self,
        market_data: MarketData,
        universe: Sequence[str],
        benchmark: str,
    ) -> None:
        self.market_data = market_data  # price source
        self.universe = list(universe)  # asset universe
        self.benchmark = benchmark  # benchmark symbol

    def run(self, target_weights: pd.DataFrame) -> BacktestResult:
        """Run a backtest given a time series of target weights.

        Parameters
        ----------
        target_weights:
            DataFrame indexed by rebalancing dates.
            Weights on each date must be finite and sum to one.
        """

        if target_weights.empty:
            raise ValueError("target_weights must not be empty")
        missing = [
            symbol
            for symbol in self.universe
            if symbol not in target_weights.columns
        ]
        if missing:
            raise ValueError(f"target_weights missing symbols: {missing}")
        supplied = target_weights.loc[:, self.universe]
        if not np.isfinite(supplied.to_numpy(dtype=float)).all():
            raise ValueError("target_weights must contain only finite values")
        if not np.allclose(supplied.sum(axis=1).to_numpy(), 1.0, atol=1e-8):
            raise ValueError("target weights must sum to one on every date")
        if target_weights.index.has_duplicates:
            raise ValueError("target_weights index must not contain duplicates")
        if not target_weights.index.is_monotonic_increasing:
            raise ValueError("target_weights index must be sorted")

        prices = self.market_data.select(
            self.universe + [self.benchmark],
        )  # price panel
        prices = prices.loc[prices.index >= target_weights.index.min()]  # trim

        # Align rebalancing dates to available price dates.
        rebal_dates = prices.index.intersection(target_weights.index)  # overlap
        if rebal_dates.empty:
            raise ValueError(
                "no overlap between price dates and target_weights index",
            )

        # Forward-fill target weights between rebalancing dates.
        tw = target_weights.reindex(prices.index, method="ffill")  # daily grid
        tw = tw.loc[:, self.universe]  # only tradable symbols

        rets = prices.pct_change(fill_method=None).dropna()  # daily returns
        asset_rets = rets[self.universe]  # asset return block
        tw_lagged = tw.shift().loc[rets.index]  # weights applied to returns
        port_rets = (asset_rets * tw_lagged).sum(
            axis=1,
            min_count=len(self.universe),
        )  # portfolio series
        bench_rets = rets[self.benchmark]  # benchmark series
        aligned = pd.concat(
            {"portfolio": port_rets, "benchmark": bench_rets},
            axis=1,
        ).dropna()

        return BacktestResult(
            dates=aligned.index,
            portfolio_returns=aligned["portfolio"],
            benchmark_returns=aligned["benchmark"],
        )
