# `assetlib` — Minimal Asset-Management Library

`assetlib` is a compact, didactic Python library used in *Python for Finance,
3rd Edition* (Chapter 21). It provides a small set of building blocks for
cross-sectional asset-management workflows: instruments and universes, market
data access, simple signals, portfolio construction helpers, a rebalancing
backtest engine, and a couple of reporting helpers.

The focus is clarity and reproducibility, not production completeness.

## Quick Start

### 1) Make the package importable

This repo keeps the library under `code/assetlib/`. To use `import assetlib`
from the project root, add `code/` to `PYTHONPATH`.

On macOS and Linux (bash/zsh):

```bash
export PYTHONPATH="$PWD/code"
```

In PowerShell on Windows:

```powershell
$env:PYTHONPATH = "$PWD\\code"
```

### 2) Load data, build a signal, tilt weights, backtest

```python
import pandas as pd

from assetlib.data import MarketData
from assetlib.signals import SignalEngine
from assetlib.signals import zscore
from assetlib.portfolio import equal_weight, signal_tilt
from assetlib.backtest import BacktestEngine
from assetlib.reporting import performance_report

universe = ["AAPL", "JPM", "TLT"]
benchmark = "SPY"

md = MarketData.load()  # default: data/eod_data.csv (project dataset)
sig = SignalEngine(market_data=md, universe=universe)

mom = sig.momentum(window=20)
z = zscore(mom)

rebalance = "ME"
raw_weights = z.resample(rebalance).last().dropna(how="any")
target_weights = raw_weights.apply(signal_tilt, axis=1)

engine = BacktestEngine(market_data=md, universe=universe, benchmark=benchmark)
result = engine.run(target_weights=target_weights)

report = performance_report(result)
print(report.table.round(4))
print(report.summary_text())
```

## Public API (at a glance)

The package root (`assetlib/__init__.py`) re-exports the main entry points:

- Core domain objects: `Instrument`, `Universe`, `Position`, `Portfolio`
- Market data helper: `MarketData`
- Signal helper: `SignalEngine`

## Package Layout

- `assetlib/core.py`
  - `Instrument`: small instrument metadata container
  - `Universe`: collection of `Instrument` objects with `.symbols` and `.get()`
  - `Position`: (symbol, quantity, price) with `.market_value`
  - `Portfolio`: holdings snapshot with derived `market_value` and `weight`
- `assetlib/data.py`
  - `MarketData.load()`: reads the EOD CSV into `.prices`
  - `.select()`: aligned price panel for a symbol list
  - `.returns()`: simple returns via `.pct_change()`
  - `.window()`: trailing window convenience
  - `.to_long()`: wide-to-long helper for reporting/joins
- `assetlib/signals.py`
  - `SignalEngine.momentum()`: rolling mean returns
  - `SignalEngine.volatility()`: rolling standard deviation of returns
  - `SignalEngine.forward_returns()`: forward return target
  - `SignalEngine.zscore()`: cross-sectional z-scores per date
  - `SignalEngine.information_coefficient()`: daily cross-sectional IC series
- `assetlib/portfolio.py`
  - `estimate_mu_sigma()`: annualized mean/cov estimation
  - `equal_weight()`: equal-weight baseline
  - `signal_tilt()`: map a cross-sectional signal into weights
  - `gmv_weights()`: global minimum-variance weights from a covariance matrix
- `assetlib/backtest.py`
  - `BacktestEngine.run()`: rebalancing backtest on target weights
  - `BacktestResult.to_frame()`: returns DataFrame with portfolio/benchmark
- `assetlib/reporting.py`
  - `performance_report()`: annualized return/vol, max drawdown, tracking error
  - `exposure_report()`: holdings snapshot view (`symbol`, `weight`, …)

## Usage Patterns

### Portfolio snapshots

Use `Portfolio` when you want a single-date view that computes weights from
quantities and prices.

```python
import pandas as pd
from assetlib.core import Portfolio

holdings = pd.DataFrame(
    {
        "symbol": ["A", "B"],
        "quantity": [10.0, 20.0],
        "price": [100.0, 50.0],
        "sector": ["Tech", "Tech"],
        "region": ["US", "US"],
    }
)
port = Portfolio(holdings)
print(port.weights)
print(port.sector_weights())
```

### Signal → weights

`signal_tilt()` expects a cross-sectional signal for one date (a `Series`).

```python
import pandas as pd
from assetlib.portfolio import signal_tilt

signal = pd.Series({"A": 0.2, "B": -0.1, "C": 0.3})
weights = signal_tilt(signal, long_only=True, cap_per_asset=0.6)
```

## Notes and Limitations

- Returns are simple returns (`.pct_change()`), and backtesting is a
  buy-and-hold-with-rebalancing abstraction without transaction costs.
- Many real-world concerns are intentionally omitted (corporate actions,
  survivorship bias, borrow costs, constraints, execution, and so on).
