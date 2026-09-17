# `engine` — Minimal Market/Broker Engine

`engine` is a small, self-contained trading engine used in *Python for Finance,
3rd Edition* (Chapter 24) and the deployment examples in Chapter 25. It models
four responsibilities:

- market data as a feed of bid/ask `Tick` objects,
- a minimal `PaperBroker` that executes market orders and tracks positions,
- a `TradingSession` event loop that ties feed, broker, and strategy together,
- JSON-friendly receipts and account snapshots for inspection and logging.

The goal is a clear teaching scaffold, not a production OMS/EMS.

## Quick Start

### 1) Make the package importable

This repo keeps the library under `code/engine/`. To use `import engine` from
the project root, add `code/` to `PYTHONPATH`.

On macOS and Linux (bash/zsh):

```bash
export PYTHONPATH="$PWD/code"
```

In PowerShell on Windows:

```powershell
$env:PYTHONPATH = "$PWD\\code"
```

### 2) Run a minimal strategy callback

```python
from engine import PaperBroker, TradingSession, build_feed

def strategy(session, tick):
    if not session.receipts:
        session.place_market_order(
            symbol=tick.symbol,
            side="buy",
            quantity=10.0,
            client_order_id="ENTRY-001",
            meta={"strategy": "demo"},
        )
        session.place_stop_loss(
            symbol=tick.symbol,
            stop_price=round(tick.bid * 0.98, 6),
        )

feed = build_feed(symbol="EURUSD", mode="historical")
broker = PaperBroker(initial_cash=50_000.0, account_id="SIM-001")
session = TradingSession(feed=feed, broker=broker)

history = session.run(strategy)
print(history[["mid", "cash", "equity", "position_quantity"]].head())
print(session.receipts[-1]["event"])
```

## Public API

The package root (`engine/__init__.py`) re-exports the primary entry points:

- Data and feeds: `HistoricalFeed`, `GBMFeed`, `build_feed()`
- Market-data helpers: `estimate_gbm_parameters()`, `generate_simulated_timestamps()`
- Models: `Tick`, `OrderRequest`, `StopOrder`, `Position`
- Execution: `PaperBroker`
- Orchestration: `TradingSession`

## Package Layout

- `engine/models.py`
  - `Tick`: bid/ask update with `.mid`, `.spread`, `.to_dict()`
  - `OrderRequest`: market order request with `.to_dict()`
  - `StopOrder`: stop-loss order state with `.to_dict()`
  - `Position`: position state with `.unrealized_pnl`, `.to_dict()`
- `engine/data.py`
  - `HistoricalFeed`: replays EOD prices as pseudo-real-time ticks
  - `GBMFeed`: simulated ticks based on GBM parameters estimated from history
  - `build_feed()`: factory returning either `HistoricalFeed` or `GBMFeed`
  - `estimate_gbm_parameters()`: annualized drift/vol estimation from log returns
  - `generate_simulated_timestamps()`: fixed/jittered/random arrival times
- `engine/broker.py`
  - `PaperBroker`: market-order execution, position updates, stop-loss handling
- `engine/session.py`
  - `TradingSession`: runs the event loop and collects a history DataFrame

## Usage Patterns

### Switching between historical replay and simulation

`build_feed()` is the simplest entry point:

```python
from engine import build_feed

hist = build_feed(symbol="EURUSD", mode="historical", spread_bps=1.0)
sim = build_feed(
    symbol="EURUSD",
    mode="simulated",
    periods=100,
    seed=7,
    base_interval="1s",
    arrival_model="jittered",
    jitter_seconds=0.25,
    spread_bps=1.0,
)
```

### Receipts and account snapshots

The broker returns JSON-friendly receipts and snapshots for inspection:

- `PaperBroker.place_order()` returns an `order_filled` receipt that includes
  a nested `account_snapshot`.
- `PaperBroker.get_account_snapshot()` returns `cash`, `equity`, realized and
  unrealized P&L, open positions, and open stop orders.

### Strategy callback interface

A strategy callback has signature:

```python
def strategy(session: TradingSession, tick: Tick) -> None:
    ...
```

It can:

- read the current tick and historical state via `session.history`,
- place orders via `session.place_market_order()` and `session.place_stop_loss()`,
- store additional state in closures or class instances.

## Tests

The engine has a small test suite in `tests/test_engine.py`.

From the project root:

```bash
pytest -q
```

## Notes and Limitations

- Orders are market orders only; fills happen at bid or ask.
- The account model is a simple cash account with marked-to-mid positions.
- Stop orders are a minimal stop-loss mechanism keyed by symbol.
- The engine is single-symbol by design for readability.
