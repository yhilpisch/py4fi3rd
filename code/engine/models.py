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


def _serialize_value(value: Any) -> Any:
    """Convert selected Python objects to JSON-friendly values."""

    if isinstance(value, pd.Timestamp):
        return value.isoformat()  # stable JSON-friendly timestamp format
    if isinstance(value, dict):
        return {key: _serialize_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_serialize_value(val) for val in value]
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
        data["mid"] = round(self.mid, 6)  # expose derived mid price explicitly
        data["spread"] = round(self.spread, 6)  # derived bid-ask spread
        return _serialize_value(data)


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
        return _serialize_value(asdict(self))


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
        return _serialize_value(asdict(self))


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

    def apply_fill(
        self,
        signed_quantity: float,
        fill_price: float,
    ) -> float:
        """Apply a signed fill and return the realized P&L change."""
        old_qty = self.quantity
        new_qty = old_qty + signed_quantity
        realized_change = 0.0

        if abs(old_qty) < 1e-12:
            self.quantity = new_qty  # opening trade from flat
            self.avg_price = (
                fill_price if abs(new_qty) > 1e-12 else 0.0
            )  # initialize cost basis
            return realized_change

        if old_qty * signed_quantity > 0.0:
            total_abs = abs(old_qty) + abs(signed_quantity)  # scale-in trade
            self.avg_price = (
                abs(old_qty) * self.avg_price
                + abs(signed_quantity) * fill_price
            ) / total_abs  # weighted-average cost basis
            self.quantity = new_qty
            return realized_change

        closed_qty = min(abs(old_qty), abs(signed_quantity))  # closing leg size
        realized_change = closed_qty * (fill_price - self.avg_price)
        if old_qty < 0.0:
            realized_change *= -1.0  # short P&L has inverted sign convention

        self.realized_pnl += realized_change  # position-level realized P&L
        self.quantity = new_qty  # residual or reversed position

        if abs(new_qty) < 1e-12:
            self.quantity = 0.0  # snap tiny floats back to flat
            self.avg_price = 0.0
            self.market_price = 0.0
        elif old_qty * new_qty < 0.0:
            self.avg_price = fill_price  # reversal starts new cost basis

        return realized_change

    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.market_price - self.avg_price)

    def to_dict(self) -> dict[str, Any]:
        return _serialize_value(
            {
                "symbol": self.symbol,
                "quantity": self.quantity,
                "avg_price": self.avg_price,
                "market_price": self.market_price,  # latest mid mark
                "market_value": self.market_value,  # marked market value
                "realized_pnl": self.realized_pnl,
                "unrealized_pnl": self.unrealized_pnl,  # open profit/loss at mark
            }
        )
