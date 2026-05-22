"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 1 - Why Python for Finance.

This module collects small, production-oriented examples that echo
the major themes of Chapter 1:

- representing simple financial data in Python containers,
- performing basic numerical aggregations,
- and sketching repeatable, script-style workflows.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def total_notional(notionals: Iterable[float]) -> float:
    """Return the total notional of a collection of trades.

    The function accepts any iterable of numbers and converts it to a list
    exactly once so that the data can be iterated multiple times if needed.
    """

    amounts = list(notionals)
    return float(sum(amounts))


@dataclass
class SimpleTrade:
    """Minimal representation of a trade used for reporting examples.

    Parameters
    ----------
    symbol:
        The traded instrument, for example ``"AAPL"``.
    qty:
        The signed quantity (positive for long, negative for short).
    price:
        The execution price in the trading currency.
    """

    symbol: str
    qty: float
    price: float

    def notional(self) -> float:
        """Return the signed notional value ``qty * price``."""

        return self.qty * self.price


def portfolio_notional(trades: Iterable[SimpleTrade]) -> float:
    """Compute the aggregate notional of several trades."""

    return float(sum(t.notional() for t in trades))


def main() -> None:
    """Demonstrate the basic helpers with a tiny example."""

    trades = [
        SimpleTrade("AAPL", 10, 180.0),
        SimpleTrade("MSFT", -5, 350.0),
    ]
    total = portfolio_notional(trades)

    # Simple invariant check: long AAPL notional plus short MSFT notional.
    assert total == 10 * 180.0 - 5 * 350.0


if __name__ == "__main__":
    main()

