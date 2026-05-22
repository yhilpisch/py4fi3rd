"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 24 - Building a Market and Broker for Trading.

Session runner for the example trading engine package.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from engine.broker import PaperBroker
    from engine.models import OrderRequest, Tick
else:
    from .broker import PaperBroker
    from .models import OrderRequest, Tick


Strategy = Callable[["TradingSession", Tick], None]


class TradingSession:
    """Connect a market feed, broker, and user-defined strategy."""

    def __init__(self, feed, broker: PaperBroker) -> None:
        self.feed = feed
        self.broker = broker
        self.history: list[dict[str, object]] = []
        self.receipts: list[dict[str, object]] = []

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        client_order_id: str | None = None,
        meta: dict[str, object] | None = None,
    ) -> dict[str, object]:
        receipt = self.broker.place_order(
            OrderRequest(
                symbol=symbol,
                side=side,
                quantity=quantity,
                client_order_id=client_order_id,
                meta={} if meta is None else meta,  # keep payload JSON-friendly
            )
        )
        self.receipts.append(receipt)  # keep an execution log in memory
        return receipt

    def place_stop_loss(
        self,
        symbol: str,
        stop_price: float,
        quantity: float | None = None,
    ) -> dict[str, object]:
        receipt = self.broker.place_stop_loss(
            symbol=symbol,
            stop_price=stop_price,
            quantity=quantity,
        )
        self.receipts.append(receipt)  # keep the acceptance receipt as well
        return receipt

    def run(self, strategy: Strategy) -> pd.DataFrame:
        """Run the strategy over the feed and collect account snapshots."""

        for tick in self.feed:
            self.receipts.extend(self.broker.on_tick(tick))  # stop fills first
            strategy(self, tick)  # strategy reacts to the latest tick
            snapshot = self.broker.get_account_snapshot()  # post-strategy state
            self.history.append(
                {
                    "timestamp": tick.timestamp,
                    "symbol": tick.symbol,
                    "source": tick.source,
                    "bid": tick.bid,
                    "ask": tick.ask,
                    "mid": tick.mid,
                    "cash": snapshot["cash"],
                    "equity": snapshot["equity"],
                    "realized_pnl": snapshot["realized_pnl"],
                    "unrealized_pnl": snapshot["unrealized_pnl"],
                    "position_quantity": (
                        snapshot["positions"][0]["quantity"]
                        if snapshot["positions"] else 0.0
                    ),
                }
            )
        return pd.DataFrame(self.history).set_index("timestamp")  # event log
