"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 21 - A Small Asset Management Library in Python.

Signal and forecast-target utilities.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
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
class SignalEngine:
    """Build simple price-based signals and forecast targets."""

    market_data: MarketData
    universe: Sequence[str]

    @property
    def returns(self) -> pd.DataFrame:
        return self.market_data.returns(self.universe)  # base daily returns

    def momentum(self, window: int = 20) -> pd.DataFrame:
        rets = self.returns  # reuse cached returns
        return rets.rolling(window).mean()  # simple momentum

    def volatility(self, window: int = 60) -> pd.DataFrame:
        rets = self.returns
        return rets.rolling(window).std()  # rolling volatility

    def forward_returns(self, horizon: int = 5) -> pd.DataFrame:
        rets = self.returns
        return rets.shift(-horizon).rolling(horizon).sum()  # forward sums

    @staticmethod
    def zscore(signal_df: pd.DataFrame) -> pd.DataFrame:
        def _z(row: pd.Series) -> pd.Series:
            if row.isna().all():
                return row
            mean = row.mean()
            std = row.std(ddof=0)
            if std == 0.0:
                return pd.Series(
                    np.zeros(len(row)),
                    index=row.index,
                )  # flat when no dispersion
            return (row - mean) / std  # z-scores

        return signal_df.apply(_z, axis=1)

    @staticmethod
    def information_coefficient(
        signal_df: pd.DataFrame,
        target_df: pd.DataFrame,
        method: str = "spearman",
    ) -> pd.Series:
        """Daily cross-sectional information coefficient."""

        if signal_df.shape != target_df.shape:
            raise ValueError(
                "signal_df and target_df must have the same shape",
            )

        aligned_signal, aligned_target = signal_df.align(
            target_df,
            join="inner",
        )  # ensure aligned index/columns
        rows = []
        for date, x in aligned_signal.iterrows():
            y = aligned_target.loc[date]  # matching targets
            if x.isna().any() or y.isna().any():
                continue
            rows.append(x.rank().corr(y, method=method))  # cross-sectional IC
        return pd.Series(rows, name="ic")
