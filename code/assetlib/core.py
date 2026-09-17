"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 21 - A Small Asset Management Library in Python.

Core domain objects: instruments, universes, positions, and portfolios.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence

import pandas as pd

_ANNUALIZATION_FACTOR = 252.0  # trading days per year


@dataclass(frozen=True)
class Instrument:
    """Minimal representation of a tradable instrument."""

    symbol: str
    asset_class: str = "Equity"
    sector: Optional[str] = None
    region: Optional[str] = None
    currency: str = "USD"


class Universe:
    """Collection of instruments that defines an investable universe."""

    def __init__(self, instruments: Sequence[Instrument]) -> None:
        self._instruments: List[Instrument] = list(instruments)  # copy input
        self._by_symbol = {
            ins.symbol: ins for ins in self._instruments
        }  # index

    @property
    def instruments(self) -> List[Instrument]:
        return list(self._instruments)  # defensive copy

    @property
    def symbols(self) -> List[str]:
        return [ins.symbol for ins in self._instruments]  # symbol view

    def get(self, symbol: str) -> Instrument:
        return self._by_symbol[symbol]  # raises KeyError if missing

    @classmethod
    def from_metadata(cls, rows: Iterable[Mapping[str, object]]) -> "Universe":
        instruments = [
            Instrument(
                symbol=str(row["symbol"]),
                asset_class=str(row.get("asset_class", "Equity")),
                sector=str(row.get("sector", "")) or None,
                region=str(row.get("region", "")) or None,
                currency=str(row.get("currency", "USD")),
            )
            for row in rows
        ]
        return cls(instruments)


@dataclass
class Position:
    """Single position in a portfolio."""

    symbol: str
    quantity: float
    price: float

    @property
    def market_value(self) -> float:
        return self.quantity * self.price  # position notional


class Portfolio:
    """Cross-sectional snapshot of portfolio holdings.

    Internally represented as a DataFrame with at least:
    - symbol
    - quantity
    - price
    and derived columns:
    - market_value
    - weight
    """

    def __init__(self, holdings: pd.DataFrame) -> None:
        required = {"symbol", "quantity", "price"}
        missing = required - set(holdings.columns)
        if missing:
            raise ValueError(f"holdings missing required columns: {missing}")

        df = holdings.copy()  # avoid mutating caller
        df["market_value"] = df["quantity"] * df["price"]  # position values
        total_mv = df["market_value"].sum()  # portfolio notional
        if total_mv == 0.0:
            raise ValueError("total market value is zero")
        df["weight"] = df["market_value"] / total_mv  # normalized weights
        self._holdings = df

    @property
    def holdings(self) -> pd.DataFrame:
        return self._holdings.copy()  # defensive copy

    @property
    def weights(self) -> pd.Series:
        return self._holdings.set_index("symbol")["weight"]  # weight by symbol

    @classmethod
    def from_holdings_dataframe(cls, holdings: pd.DataFrame) -> "Portfolio":
        return cls(holdings=holdings)

    def sector_weights(self) -> pd.Series:
        if "sector" not in self._holdings.columns:
            raise ValueError("holdings do not contain a 'sector' column")
        return self._holdings.groupby("sector")["weight"].sum()  # sector sums

    def region_weights(self) -> pd.Series:
        if "region" not in self._holdings.columns:
            raise ValueError("holdings do not contain a 'region' column")
        return self._holdings.groupby("region")["weight"].sum()  # region sums

    def top_concentrations(self, n: int = 10) -> pd.DataFrame:
        """Return top-n positions by weight."""
        return self._holdings.sort_values(
            "weight",
            ascending=False,
        ).head(n)  # largest weights first
