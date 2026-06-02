"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 4 - Data Types and Structures.

The short snippets in Chapter 4 focus on individual language features.
This module pulls several of those ideas together into small, reusable
utilities with explicit type hints and docstrings:

- precise decimal arithmetic for monetary amounts,
- simple container transformations,
- and safe dictionary access helpers.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, getcontext
from typing import Any


def apply_fee(amount: Decimal, fee_rate: Decimal) -> Decimal:
    """Return ``amount`` minus a proportional fee, quantized to cents."""

    getcontext().prec = 16
    fee = amount * fee_rate
    net = amount - fee
    return net.quantize(Decimal("0.01"))


def to_spreads(quotes: Iterable[tuple[float, float]]) -> list[float]:
    """Compute bid/ask spreads from an iterable of ``(bid, ask)`` quotes."""

    return [ask - bid for bid, ask in quotes]


@dataclass
class Quote:
    """Simple quote container that enforces presence of bid/ask fields."""

    symbol: str
    bid: float
    ask: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Quote":
        """Construct a quote from a mapping.

        Raises ``KeyError`` if required fields are missing.
        """

        return cls(
            symbol=str(data["symbol"]),
            bid=float(data["bid"]),
            ask=float(data["ask"]),
        )

    @property
    def mid(self) -> float:
        """Return the mid price ``(bid + ask) / 2``."""

        return 0.5 * (self.bid + self.ask)


def main() -> None:
    """Run a few self-checks so the module can be executed directly."""

    net = apply_fee(Decimal("100.00"), Decimal("0.0015"))
    assert net == Decimal("99.85")

    spreads = to_spreads([(1.0810, 1.0812), (1.0820, 1.0823)])
    assert all(s > 0 for s in spreads)

    q = Quote.from_mapping({"symbol": "EURUSD", "bid": 1.0810, "ask": 1.0812})
    assert q.mid > q.bid


if __name__ == "__main__":
    main()
