"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 24 - Building a Market and Broker for Trading.

Core data models for the example trading engine package.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


def _serialise_value(value: Any) -> Any:
    """Convert selected Python objects to JSON-friendly values."""

    if isinstance(value, pd.Timestamp):
        return value.isoformat()  # stable JSON-friendly timestamp format
    if isinstance(value, dict):
        return {key: _serialise_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_serialise_value(val) for val in value]
    return value


@dataclass(slots=True)
class Tick:
    """Single bid-ask update emitted by a market feed."""

    timestamp: pd.Timestamp
    symbol: str
    bid: float
    ask: float
    bid_size: float | None = None
    ask_size: float | None = None
    source: str = "historical"

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)  # start from dataclass fields only
        data["mid"] = self.mid  # expose derived mid price explicitly
        data["spread"] = self.spread  # expose current bid-ask spread
        return _serialise_value(data)


@dataclass(slots=True)
class OrderRequest:
    """User-facing order request."""

    symbol: str
    side: str
    quantity: float
    order_type: str = "market"
    stop_price: float | None = None
    client_order_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialise_value(asdict(self))


@dataclass(slots=True)
class StopOrder:
    """Simple stop-loss order maintained by the broker."""

    order_id: str
    symbol: str
    side: str
    quantity: float
    stop_price: float
    status: str = "working"

    def to_dict(self) -> dict[str, Any]:
        return _serialise_value(asdict(self))


@dataclass(slots=True)
class Position:
    """Position state for a single symbol."""

    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    market_price: float = 0.0
    realized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.market_price

    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.market_price - self.avg_price)

    def to_dict(self) -> dict[str, Any]:
        return _serialise_value(
            {
                "symbol": self.symbol,
                "quantity": self.quantity,
                "avg_price": self.avg_price,
                "market_price": self.market_price,  # latest mid mark
                "market_value": self.market_value,  # marked market value
                "realized_pnl": self.realized_pnl,
                "unrealized_pnl": self.unrealized_pnl,  # open P&L at mark
            }
        )
