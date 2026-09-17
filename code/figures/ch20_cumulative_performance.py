"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 20 - Asset Management Systems and Reporting.

Figure: Cumulative portfolio and benchmark performance.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import pathlib

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def load_prices_and_holdings() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load prices and the small holdings snapshot."""

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    local = base_dir / "data" / "eod_data.csv"
    remote = "https://hilpisch.com/eod_data.csv"
    source: str | pathlib.Path = local if local.exists() else remote

    prices = pd.read_csv(source, parse_dates=["Date"], index_col="Date")

    holdings = pd.DataFrame(
        {
            "symbol": ["AAPL", "NVDA", "JPM", "TLT"],
            "quantity": [120, 80, 150, 1000],
        }
    )
    return prices, holdings


def compute_cumulative_returns(
    prices: pd.DataFrame, holdings: pd.DataFrame, window_days: int = 2 * 252
) -> tuple[pd.Series, pd.Series]:
    """Compute cumulative portfolio and benchmark returns."""

    symbols = holdings["symbol"].tolist()
    sub = prices[symbols + ["SPY"]].dropna(how="any")
    if len(sub) > window_days:
        sub = sub.iloc[-window_days:]

    quantities = holdings.set_index("symbol")["quantity"]
    values = sub.mul(quantities, axis=1)
    port_val = values.sum(axis=1)

    r_port = port_val.pct_change().dropna()
    r_bench = sub["SPY"].pct_change().dropna()

    aligned = pd.concat(
        {"portfolio": r_port, "benchmark": r_bench},
        axis=1,
    ).dropna()

    cum_port = (1.0 + aligned["portfolio"]).cumprod()
    cum_bench = (1.0 + aligned["benchmark"]).cumprod()
    return cum_port, cum_bench


def main() -> None:
    """Generate the cumulative performance comparison figure."""

    mpl.use("Agg", force=True)
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

    prices, holdings = load_prices_and_holdings()
    cum_port, cum_bench = compute_cumulative_returns(prices, holdings)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cum_port.index, cum_port.values, label="Portfolio")
    ax.plot(cum_bench.index, cum_bench.values, label="Benchmark (SPY)")
    ax.set_ylabel("Cumulative value (start = 1.0)")
    ax.set_title("Cumulative portfolio vs benchmark performance")
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch20_cumulative_performance.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
