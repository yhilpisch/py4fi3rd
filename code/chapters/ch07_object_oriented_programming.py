"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 7 - Object-Oriented Programming.

This module collects slightly more structured, reusable implementations
of the OOP patterns introduced in Chapter 7:

- position and portfolio classes with explicit methods,
- a clear FX position model with reporting currency,
- and a small valuation engine that uses a price source interface.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable


@dataclass
class Position:
    """Basic financial position with a symbol, quantity, and price."""

    symbol: str
    qty: float
    price: float  # price in local currency

    def market_value(self) -> float:
        """Return the position's market value in local currency."""

        return self.qty * self.price


@dataclass
class FXPosition(Position):
    """Position converted from local to reporting currency."""

    reporting_ccy: str
    fx_to_reporting: float

    def market_value(self) -> float:
        """Return the position's market value in the reporting currency."""

        local_value = super().market_value()
        return local_value * self.fx_to_reporting


class PriceSource(ABC):
    """Abstract interface for objects that can provide prices."""

    @abstractmethod
    def get_price(self, symbol: str) -> float:
        """Return the latest price for *symbol*."""


class DictPriceSource(PriceSource):
    """Simple in-memory price source backed by a dictionary."""

    def __init__(self, prices: dict[str, float]) -> None:
        self._prices = prices

    def get_price(self, symbol: str) -> float:
        return self._prices[symbol]


class Portfolio:
    """Collection of positions with convenience methods."""

    def __init__(self, positions: Iterable[Position]) -> None:
        self.positions = list(positions)

    def total_value(self) -> float:
        """Return total market value across all positions."""

        return float(sum(p.market_value() for p in self.positions))

    def value_by_symbol(self, symbol: str) -> float:
        """Return the aggregated market value for a given symbol."""

        return float(
            sum(p.market_value() for p in self.positions if p.symbol == symbol)
        )


class ValuationEngine:
    """Create positions by pulling prices from an injected price source."""

    def __init__(self, price_source: PriceSource, reporting_ccy: str) -> None:
        self.price_source = price_source
        self.reporting_ccy = reporting_ccy

    def price_position(self, symbol: str, qty: float) -> Position:
        """Return a ``Position`` instance for *symbol* and *qty*."""

        price = self.price_source.get_price(symbol)
        return Position(symbol, qty, price)


def main() -> None:
    """Run a short example portfolio and basic invariants."""

    src = DictPriceSource({"AAPL": 180.0, "BOND_EUR": 1.10})
    engine = ValuationEngine(src, reporting_ccy="USD")

    pos_equity = engine.price_position("AAPL", 10)
    pos_bond = FXPosition(
        "BOND_EUR",
        100_000,
        price=1.10,
        reporting_ccy="USD",
        fx_to_reporting=1.08,
    )

    portfolio = Portfolio([pos_equity, pos_bond])
    total = portfolio.total_value()
    assert total > 0.0
    assert portfolio.value_by_symbol("AAPL") == pos_equity.market_value()


if __name__ == "__main__":
    main()
