"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 21 - A Small Asset Management Library in Python.

Reporting utilities for performance, risk, and exposures.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from assetlib.backtest import BacktestResult
    from assetlib.core import Portfolio
else:
    from .backtest import BacktestResult
    from .core import Portfolio


def _annualization_factor() -> float:
    return 252.0  # trading days per year


@dataclass
class PerformanceReport:
    table: pd.DataFrame

    def summary_text(self) -> str:
        port = self.table.loc["Portfolio"]  # portfolio row
        bench = self.table.loc["Benchmark"]  # benchmark row
        msg = (
            "Over the most recent window, the portfolio delivered an "
            f"annualized return of {port['annualized_return']:.1%} versus "
            f"{bench['annualized_return']:.1%} for the benchmark, with "
            f"an annualized volatility of {port['annualized_volatility']:.1%} "
            f"and a maximum drawdown of {port['max_drawdown']:.1%}."
        )
        return msg


def performance_report(result: BacktestResult) -> PerformanceReport:
    """Build a compact performance and risk report."""

    r_port = result.portfolio_returns  # portfolio returns
    r_bench = result.benchmark_returns  # benchmark returns
    ann_factor = _annualization_factor()  # annualization factor
    n_obs = len(r_port)  # number of observations

    port_total = (1.0 + r_port).prod()  # cumulative growth
    bench_total = (1.0 + r_bench).prod()
    port_ann = port_total ** (ann_factor / n_obs) - 1.0  # annualized return
    bench_ann = bench_total ** (ann_factor / n_obs) - 1.0

    port_vol = r_port.std(ddof=1) * np.sqrt(ann_factor)  # annualized vol
    bench_vol = r_bench.std(ddof=1) * np.sqrt(ann_factor)

    active = r_port - r_bench  # active returns
    te_ann = active.std(ddof=1) * np.sqrt(ann_factor)  # tracking error

    def max_drawdown(returns: pd.Series) -> float:
        cum = (1.0 + returns).cumprod()
        running_max = cum.cummax()
        drawdowns = cum / running_max - 1.0
        return float(drawdowns.min())

    report = pd.DataFrame(
        {
            "annualized_return": [port_ann, bench_ann],
            "annualized_volatility": [port_vol, bench_vol],
            "max_drawdown": [max_drawdown(r_port), max_drawdown(r_bench)],
            "annualized_tracking_error": [te_ann, 0.0],
        },
        index=["Portfolio", "Benchmark"],
    )

    if te_ann != 0.0:
        ir = (port_ann - bench_ann) / te_ann  # information ratio
    else:
        ir = np.nan
    report["information_ratio"] = ir
    return PerformanceReport(table=report)


def exposure_report(portfolio: Portfolio) -> pd.DataFrame:
    """Minimal holdings and sector/region exposure report."""

    holdings = portfolio.holdings  # snapshot
    return holdings[
        ["symbol", "quantity", "price", "market_value", "weight"]
    ]  # key columns
