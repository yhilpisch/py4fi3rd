"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 20 - Asset Management Systems and Reporting.

Companion code for the core examples in Chapter 20:

- holdings and exposure snapshots
- simple performance and risk report vs a benchmark
- sector-level contribution estimates over a recent window

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_EOD = ROOT / "data" / "eod_data.csv"


@dataclass
class PortfolioInputs:
    """Container for portfolio, benchmark, and holdings inputs."""

    returns_port: pd.Series
    returns_bench: pd.Series
    holdings: pd.DataFrame


def load_prices() -> pd.DataFrame:
    """Load end-of-day prices from the local CSV or remote fallback."""

    if DATA_EOD.exists():
        source: str | Path = DATA_EOD
    else:
        source = "https://hilpisch.com/eod_data.csv"

    prices = pd.read_csv(source, parse_dates=["Date"], index_col="Date")
    return prices.dropna(how="any")


def holdings_snapshot() -> pd.DataFrame:
    """Return the small holdings snapshot used in Chapter 20."""

    holdings = pd.DataFrame(
        {
            "symbol": ["AAPL", "NVDA", "JPM", "SPY"],
            "quantity": [120, 80, 150, 200],
            "price": [180.25, 820.10, 145.30, 520.10],
            "sector": [
                "Technology",
                "Technology",
                "Financials",
                "Equity Index",
            ],
            "region": ["US", "US", "US", "Global"],
            "currency": ["USD", "USD", "USD", "USD"],
        }
    )
    holdings["market_value"] = holdings["quantity"] * holdings["price"]
    total_value = float(holdings["market_value"].sum())
    holdings["weight"] = holdings["market_value"] / total_value
    return holdings


def prepare_portfolio_inputs(
    window_days: int = 2 * 252,
) -> PortfolioInputs:
    """Prepare portfolio and benchmark return series and holdings."""

    prices = load_prices()
    holdings = holdings_snapshot()
    symbols = holdings["symbol"].tolist()

    sub = prices[symbols].dropna(how="any")
    if len(sub) > window_days:
        sub = sub.iloc[-window_days:]

    values = sub.mul(holdings.set_index("symbol")["quantity"], axis=1)
    port_val = values.sum(axis=1)
    r_port = port_val.pct_change().dropna()

    r_bench = sub["SPY"].pct_change().dropna()
    aligned = pd.concat(
        {"portfolio": r_port, "benchmark": r_bench},
        axis=1,
    ).dropna()

    returns_port = aligned["portfolio"]
    returns_bench = aligned["benchmark"]
    return PortfolioInputs(returns_port, returns_bench, holdings)


def max_drawdown(returns: pd.Series) -> float:
    """Compute max drawdown from a return series."""

    cum = (1.0 + returns).cumprod()
    running_max = cum.cummax()
    drawdowns = cum / running_max - 1.0
    return float(drawdowns.min())


def performance_report(
    returns_port: pd.Series,
    returns_bench: pd.Series,
) -> pd.DataFrame:
    """Compute a compact performance and risk report."""

    ann_factor = 252.0
    n_obs = float(len(returns_port))

    # Annualized geometric returns
    port_total = float((1.0 + returns_port).prod())
    bench_total = float((1.0 + returns_bench).prod())
    port_ann = port_total ** (ann_factor / n_obs) - 1.0
    bench_ann = bench_total ** (ann_factor / n_obs) - 1.0

    # Annualized volatility
    port_vol = float(returns_port.std(ddof=1) * np.sqrt(ann_factor))
    bench_vol = float(returns_bench.std(ddof=1) * np.sqrt(ann_factor))

    # Annualized tracking error and information ratio
    active = returns_port - returns_bench
    te_ann = float(active.std(ddof=1) * np.sqrt(ann_factor))
    ir = port_ann - bench_ann
    if te_ann > 0.0:
        ir = ir / te_ann
    else:
        ir = float("nan")

    row_port = {
        "annualized_return": port_ann,
        "annualized_volatility": port_vol,
        "max_drawdown": max_drawdown(returns_port),
    }
    row_bench = {
        "annualized_return": bench_ann,
        "annualized_volatility": bench_vol,
        "max_drawdown": max_drawdown(returns_bench),
    }
    report = pd.DataFrame(
        [row_port, row_bench],
        index=["Portfolio", "Benchmark"],
    )
    report["annualized_tracking_error"] = [te_ann, 0.0]
    report["information_ratio"] = [ir, np.nan]
    return report


def sector_contributions(
    returns_port: pd.Series,
    holdings: pd.DataFrame,
    prices: pd.DataFrame | None = None,
) -> Tuple[pd.Series, pd.Series]:
    """Estimate sector weights and contributions over the sample window.

    Returns a pair of Series:

    - average sector weights
    - cumulative sector contributions to portfolio return
    """

    if prices is None:
        prices = load_prices()

    symbols = holdings["symbol"].tolist()
    sub = prices[symbols].dropna(how="any")
    rets_assets = sub.pct_change().dropna()

    values = sub.mul(holdings.set_index("symbol")["quantity"], axis=1)
    values = values.loc[rets_assets.index]
    weights = values.div(values.sum(axis=1), axis=0)

    sector_map = holdings.set_index("symbol")["sector"]
    sector_weights_time = weights.T.groupby(sector_map).sum().T
    sector_weights = sector_weights_time.mean()

    contrib = (weights * rets_assets).sum()
    sector_contrib = contrib.groupby(sector_map).sum()

    total_port_return = float((1.0 + returns_port).prod() - 1.0)
    if total_port_return != 0.0:
        sector_contrib = sector_contrib / total_port_return

    return (
        sector_weights.sort_values(ascending=False),
        sector_contrib.sort_values(ascending=False),
    )


def main() -> None:
    """Run a compact Chapter 20 demo and print key reports."""

    inputs = prepare_portfolio_inputs()
    report = performance_report(inputs.returns_port, inputs.returns_bench)

    print("== Holdings snapshot ==")
    print(inputs.holdings)
    print()

    print("== Performance and risk report (annualized) ==")
    print(report.round(3))
    print()

    prices = load_prices()
    sector_w, sector_c = sector_contributions(
        inputs.returns_port, inputs.holdings, prices
    )

    print("== Average sector weights over the sample window ==")
    print(sector_w.round(3))
    print()

    print("== Sector contributions as share of total return ==")
    print(sector_c.round(3))
    print()


if __name__ == "__main__":
    main()
