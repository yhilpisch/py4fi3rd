"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 24 - Building a Market and Broker for Trading.

Paper broker implementation for the example trading engine package.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from itertools import count
from typing import Any

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from engine.models import OrderRequest, Position, StopOrder, Tick
else:
    from .models import OrderRequest, Position, StopOrder, Tick


def _round_floats(value: Any, digits: int = 6) -> Any:
    """Round floats recursively to a fixed precision for stable output."""

    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: _round_floats(val, digits) for key, val in value.items()}
    if isinstance(value, list):
        return [_round_floats(val, digits) for val in value]
    return value


class PaperBroker:
    """Minimal broker that executes market orders on incoming ticks."""

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        account_id: str = "SIM-001",
    ):
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.account_id = account_id
        self.positions: dict[str, Position] = {}
        self.stop_orders: dict[str, StopOrder] = {}
        self.last_tick: dict[str, Tick] = {}
        self._realized_pnl = 0.0
        self._order_ids = count(1)

    def on_tick(self, tick: Tick) -> list[dict[str, Any]]:
        """Update marks and trigger eligible stop-loss orders."""

        self.last_tick[tick.symbol] = tick  # newest quote per symbol
        position = self.positions.get(tick.symbol)
        if position is not None:
            position.market_price = tick.mid  # positions are marked at mid

        receipts: list[dict[str, Any]] = []
        stop_order = self.stop_orders.get(tick.symbol)
        if stop_order is None or stop_order.status != "working":
            return receipts

        if stop_order.side == "sell" and tick.bid <= stop_order.stop_price:
            receipts.append(
                self._execute_market_order(
                    OrderRequest(
                        symbol=tick.symbol,
                        side="sell",
                        quantity=stop_order.quantity,
                        client_order_id=stop_order.order_id,
                        meta={"trigger": "stop_loss"},
                    ),
                    fill_price=tick.bid,  # long stop exits at the bid
                    timestamp=tick.timestamp,
                )
            )
            stop_order.status = "triggered"
            self.stop_orders.pop(tick.symbol, None)
        elif stop_order.side == "buy" and tick.ask >= stop_order.stop_price:
            receipts.append(
                self._execute_market_order(
                    OrderRequest(
                        symbol=tick.symbol,
                        side="buy",
                        quantity=stop_order.quantity,
                        client_order_id=stop_order.order_id,
                        meta={"trigger": "stop_loss"},
                    ),
                    fill_price=tick.ask,  # short stop exits at the ask
                    timestamp=tick.timestamp,
                )
            )
            stop_order.status = "triggered"
            self.stop_orders.pop(tick.symbol, None)

        return receipts

    def place_order(self, order: OrderRequest) -> dict[str, Any]:
        """Validate and execute a market order."""

        if order.order_type != "market":
            raise NotImplementedError("only market orders are implemented.")
        if order.side not in {"buy", "sell"}:
            raise ValueError("side must be either 'buy' or 'sell'.")
        if order.quantity <= 0.0:
            raise ValueError("quantity must be positive.")

        tick = self.last_tick.get(order.symbol)
        if tick is None:
            msg = f"no market data received yet for {order.symbol!r}."
            raise ValueError(msg)

        fill_price = (
            tick.ask if order.side == "buy" else tick.bid
        )  # side-aware fill
        return self._execute_market_order(
            order=order,
            fill_price=fill_price,
            timestamp=tick.timestamp,
        )

    def place_stop_loss(
        self,
        symbol: str,
        stop_price: float,
        quantity: float | None = None,
    ) -> dict[str, Any]:
        """Register a stop-loss order against the current position."""

        position = self.positions.get(symbol)
        if position is None or abs(position.quantity) < 1e-12:
            raise ValueError(f"no open position for {symbol!r}.")

        side = "sell" if position.quantity > 0.0 else "buy"
        stop_quantity = (
            abs(position.quantity)
            if quantity is None
            else float(quantity)
        )  # default to full position size
        order_id = f"STOP-{next(self._order_ids):06d}"

        stop_order = StopOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=stop_quantity,
            stop_price=float(stop_price),
        )
        self.stop_orders[symbol] = stop_order
        return {
            "account_id": self.account_id,
            "event": "stop_order_accepted",
            "order": stop_order.to_dict(),
        }

    def get_positions(self) -> list[dict[str, Any]]:
        """Return active positions as JSON-friendly dictionaries."""

        active = [
            _round_floats(position.to_dict())
            for position in self.positions.values()
            if abs(position.quantity) > 1e-12  # ignore flat positions
        ]
        return sorted(active, key=lambda item: item["symbol"])

    def get_account_snapshot(self) -> dict[str, Any]:
        """Return cash, equity, and P&L information."""

        positions_value = sum(
            position.market_value
            for position in self.positions.values()
            if abs(position.quantity) > 1e-12
        )  # aggregate marked market value
        unrealized = sum(
            position.unrealized_pnl
            for position in self.positions.values()
            if abs(position.quantity) > 1e-12
        )  # aggregate open P&L
        equity = self.cash + positions_value  # simple cash account model

        return _round_floats(
            {
                "account_id": self.account_id,
                "cash": self.cash,
                "equity": equity,
                "realized_pnl": self._realized_pnl,
                "unrealized_pnl": unrealized,
                "positions": self.get_positions(),
                "open_stop_orders": [
                    order.to_dict()
                    for order in self.stop_orders.values()
                    if order.status == "working"
                ],
            }
        )

    def _execute_market_order(
        self,
        order: OrderRequest,
        fill_price: float,
        timestamp,
    ) -> dict[str, Any]:
        signed_quantity = (
            order.quantity if order.side == "buy" else -order.quantity
        )  # long positive, short negative
        self.cash -= signed_quantity * fill_price  # trade cash settlement

        position = self.positions.get(order.symbol)
        if position is None:
            position = Position(symbol=order.symbol)
            self.positions[order.symbol] = position

        realized_change = position.apply_fill(
            signed_quantity=signed_quantity,
            fill_price=float(fill_price),
        )
        self._realized_pnl += realized_change  # broker-level realized P&L
        position.market_price = self.last_tick[order.symbol].mid  # refresh mark

        return _round_floats(
            {
                "account_id": self.account_id,
                "event": "order_filled",
                "timestamp": timestamp.isoformat(),
                "order_id": (
                    order.client_order_id or f"ORD-{next(self._order_ids):06d}"
                ),
                "order": order.to_dict(),
                "fill": {
                    "price": float(fill_price),
                    "quantity": float(order.quantity),
                    "side": order.side,
                },
                "realized_pnl_change": realized_change,
                "account_snapshot": self.get_account_snapshot(),
            }
        )
