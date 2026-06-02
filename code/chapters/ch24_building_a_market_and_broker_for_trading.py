"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 24 - Building a Market and Broker for Trading.

Companion code for the chapter 24 engine scaffold.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from pprint import pprint
import sys
from types import ModuleType

import pandas as pd


SYMBOL = "EURUSD"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_engine_module() -> ModuleType:
    """Import the local engine package in script and notebook contexts."""

    try:
        return importlib.import_module("code.engine")
    except ModuleNotFoundError:
        return importlib.import_module("engine")


ENGINE = _load_engine_module()
OrderRequest = ENGINE.OrderRequest
PaperBroker = ENGINE.PaperBroker
Tick = ENGINE.Tick
TradingSession = ENGINE.TradingSession
build_feed = ENGINE.build_feed
estimate_gbm_parameters = ENGINE.estimate_gbm_parameters


def demo_strategy(session: TradingSession, tick: Tick) -> None:
    """Open one position once and attach a stop loss."""

    if not session.receipts:
        session.place_market_order(
            symbol=tick.symbol,
            side="buy",
            quantity=10.0,
            client_order_id="ENTRY-001",
            meta={"strategy": "chapter24_demo"},
        )
        session.place_stop_loss(
            symbol=tick.symbol,
            stop_price=round(tick.bid * 0.98, 6),
        )


def build_walkthrough_objects() -> dict[str, object]:
    """Create deterministic objects for the chapter walkthrough."""

    feed = build_feed(symbol=SYMBOL, mode="historical")
    hist_feed = build_feed(symbol=SYMBOL, mode="historical")
    sim_feed = build_feed(symbol=SYMBOL, mode="simulated", periods=10, seed=42)

    first_hist_tick = next(iter(hist_feed))
    first_sim_tick = next(iter(sim_feed))

    prices = pd.Series([100.0, 101.0, 102.5, 101.5, 103.0])
    mu, sigma = estimate_gbm_parameters(prices)

    broker = PaperBroker(initial_cash=50_000.0)
    broker.on_tick(first_hist_tick)
    receipt = broker.place_order(
        OrderRequest(
            symbol=SYMBOL,
            side="buy",
            quantity=10.0,
            client_order_id="ENTRY-001",
            meta={"strategy": "demo"},
        )
    )
    snapshot = broker.get_account_snapshot()
    stop_receipt = broker.place_stop_loss(
        symbol=SYMBOL,
        stop_price=round(first_hist_tick.bid * 0.98, 6),
    )
    stop_snapshot = broker.get_account_snapshot()
    trigger_tick = Tick(
        timestamp=first_hist_tick.timestamp + pd.offsets.BusinessDay(1),
        symbol=SYMBOL,
        bid=round(first_hist_tick.bid * 0.97, 6),
        ask=round(first_hist_tick.ask * 0.97, 6),
        source="synthetic_trigger",
    )
    triggered = broker.on_tick(trigger_tick)

    session = TradingSession(
        feed=feed,
        broker=PaperBroker(initial_cash=50_000.0),
    )
    session_history = session.run(demo_strategy)

    return {
        "feed": feed,
        "hist_feed": hist_feed,
        "sim_feed": sim_feed,
        "first_hist_tick": first_hist_tick,
        "first_sim_tick": first_sim_tick,
        "mu_sigma": (mu, sigma),
        "broker": broker,
        "receipt": receipt,
        "snapshot": snapshot,
        "stop_receipt": stop_receipt,
        "stop_snapshot": stop_snapshot,
        "trigger_tick": trigger_tick,
        "triggered": triggered,
        "session_history": session_history,
        "session": session,
    }


def main() -> None:
    data = build_walkthrough_objects()

    assert data["receipt"]["event"] == "order_filled"
    assert data["triggered"][0]["order"]["meta"]["trigger"] == "stop_loss"
    assert not data["broker"].get_positions()
    assert not data["session_history"].empty

    print("SECTION: imports")
    print("feed =", data["feed"])
    print("broker =", PaperBroker(initial_cash=50_000.0))
    print(
        "session =",
        TradingSession(feed=data["hist_feed"], broker=PaperBroker()),
    )

    print("\nSECTION: first_ticks")
    pprint(data["first_hist_tick"].to_dict())
    print(data["first_sim_tick"].source)

    print("\nSECTION: mu_sigma")
    print(tuple(round(value, 4) for value in data["mu_sigma"]))

    print("\nSECTION: receipt_event")
    print(data["receipt"]["event"])
    pprint(data["receipt"]["fill"])
    pprint(data["receipt"]["order"])

    print("\nSECTION: snapshot")
    pprint(data["snapshot"]["positions"])
    print(
        round(data["snapshot"]["cash"], 6),
        round(data["snapshot"]["equity"], 6),
    )
    pprint(data["snapshot"]["positions"])
    pprint(data["snapshot"]["open_stop_orders"])

    print("\nSECTION: stop")
    print(data["stop_receipt"]["event"])
    pprint(data["stop_snapshot"]["open_stop_orders"])
    pprint(data["triggered"][0]["order"]["meta"])
    pprint(data["broker"].get_positions())

    print("\nSECTION: session_history")
    cols = ["bid", "ask", "mid", "cash", "equity", "position_quantity"]
    print(data["session_history"][cols].head())

    print("\nSECTION: json")
    print(json.dumps(data["receipt"], indent=2))


if __name__ == "__main__":
    main()
