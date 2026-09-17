"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 21 - A Small Asset Management Library in Python.

This companion script exercises the assetlib package end-to-end:
domain objects, market data, signals, portfolio construction,
and backtesting.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UNIVERSE_SYMBOLS = ("AAPL", "NVDA", "JPM", "TLT")
BENCHMARK = "SPY"
MOMENTUM_WINDOW = 20
REBALANCE_FREQ = "W-FRI"


def main() -> None:
    from pathlib import Path
    import sys

    base_dir = Path(__file__).resolve().parents[2] / "code"
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))

    from assetlib.core import Instrument, Universe, Portfolio
    from assetlib.data import MarketData
    from assetlib.signals import SignalEngine
    from assetlib.portfolio import signal_tilt
    from assetlib.backtest import BacktestEngine
    from assetlib.reporting import exposure_report

    # --- Instruments and universe ---
    instruments = [
        Instrument(symbol="AAPL", sector="Technology"),
        Instrument(symbol="NVDA", sector="Technology"),
        Instrument(symbol="JPM", sector="Financials"),
        Instrument(symbol="TLT", sector="Fixed Income"),
    ]
    universe = Universe(instruments)

    # --- Market data ---
    data = MarketData.load()
    symbols = [s for s in UNIVERSE_SYMBOLS if s in data.prices.columns]
    assert len(symbols) >= 2, "need at least two symbols to proceed"

    # --- Signals ---
    engine = SignalEngine(data, universe=symbols)
    mom = engine.momentum(window=MOMENTUM_WINDOW)

    def _build_weights(signal_row):
        return signal_tilt(signal_row, long_only=True)

    w_raw = mom.dropna(how="all").apply(_build_weights, axis=1)
    w_dated = w_raw.dropna(how="all")
    w_weekly = w_dated.resample(REBALANCE_FREQ).last()
    w_weekly = w_weekly[~w_weekly.index.duplicated(keep="first")]
    assert len(w_weekly) > 0, "weekly weight grid must not be empty"

    # --- Backtest ---
    engine_bt = BacktestEngine(
        market_data=data,
        universe=symbols,
        benchmark=BENCHMARK,
    )
    result = engine_bt.run(w_weekly)

    # Compute performance from the backtest result directly.
    ann = 252.0
    r_p = result.portfolio_returns
    r_b = result.benchmark_returns
    if isinstance(r_p, pd.DataFrame):
        r_p = r_p.iloc[:, 0]
    if isinstance(r_b, pd.DataFrame):
        r_b = r_b.iloc[:, 0]
    r_p = r_p[~r_p.index.duplicated(keep="first")]
    r_b = r_b[~r_b.index.duplicated(keep="first")]

    port_ann = float((1.0 + r_p).prod() ** (ann / len(r_p)) - 1.0)
    bench_ann = float((1.0 + r_b).prod() ** (ann / len(r_b)) - 1.0)
    port_vol = float(r_p.std(ddof=1) * np.sqrt(ann))

    report = pd.DataFrame(
        {
            "annualized_return": [port_ann, bench_ann],
            "annualized_volatility": [
                port_vol,
                float(r_b.std(ddof=1) * np.sqrt(ann)),
            ],
        },
        index=["Portfolio", "Benchmark"],
    )
    print("== Performance report ==")
    print(report.round(4))
    print()

    # Build a representative portfolio snapshot from the first weight row.
    holdings_rows = []
    first_date = w_dated.index[0]
    for sym in symbols:
        price = data.prices.loc[first_date, sym]
        holdings_rows.append(
            {
                "symbol": sym,
                "quantity": 1.0,
                "price": float(price),
                "sector": universe.get(sym).sector or "",
            }
        )

    port = Portfolio(pd.DataFrame(holdings_rows))
    exp = exposure_report(port)
    print("== Exposure report (snapshot) ==")
    print(exp.round(3))

    # Self-checks.
    assert abs(float(port.weights.sum()) - 1.0) < 1e-6, "weights must sum to 1"
    assert len(report) == 2, "report must have Portfolio and Benchmark rows"
    assert report.loc["Portfolio", "annualized_volatility"] >= 0.0, (
        "volatility must be non-negative"
    )


if __name__ == "__main__":
    main()
