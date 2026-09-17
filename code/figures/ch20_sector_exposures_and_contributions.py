"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 20 - Asset Management Systems and Reporting.

Figure: Sector weights and relative return contributions.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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
            "sector": [
                "Technology",
                "Technology",
                "Financials",
                "Fixed Income",
            ],
        }
    )
    return prices, holdings


def sector_weights_and_contributions(
    prices: pd.DataFrame, holdings: pd.DataFrame, window_days: int = 2 * 252
) -> tuple[pd.Series, pd.Series]:
    """Compute average sector weights and scaled contributions."""

    symbols = holdings["symbol"].tolist()
    sub = prices[symbols].dropna(how="any")
    if len(sub) > window_days:
        sub = sub.iloc[-window_days:]

    rets_assets = sub.pct_change().dropna()

    quantities = holdings.set_index("symbol")["quantity"]
    values = sub.mul(quantities, axis=1)
    values = values.loc[rets_assets.index]
    weights = values.div(values.sum(axis=1), axis=0)

    sector_map = holdings.set_index("symbol")["sector"]
    sector_weights_time = weights.T.groupby(sector_map).sum().T
    sector_weights = sector_weights_time.mean()

    contrib = (weights * rets_assets).sum()
    sector_contrib = contrib.groupby(sector_map).sum()

    r_port = (weights * rets_assets).sum(axis=1)
    total_port_return = float((1.0 + r_port).prod() - 1.0)
    if total_port_return != 0.0:
        sector_contrib = sector_contrib / total_port_return

    return (
        sector_weights.sort_values(ascending=False),
        sector_contrib.sort_values(ascending=False),
    )


def main() -> None:
    """Generate the sector weights and contributions figure."""

    mpl.use("Agg", force=True)
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

    prices, holdings = load_prices_and_holdings()
    sector_w, sector_c = sector_weights_and_contributions(prices, holdings)

    sectors = sector_w.index.tolist()
    x = np.arange(len(sectors))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True)

    axes[0].bar(x, sector_w.values, color="tab:blue")
    axes[0].set_ylabel("Average sector weight")
    axes[0].set_title("Average sector weights")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.3)

    axes[1].bar(x, sector_c.values, color="tab:green")
    axes[1].set_ylabel("Relative contribution")
    axes[1].set_title("Sector contributions to performance")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.3)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(sectors, rotation=20, ha="right")

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch20_sector_exposures_and_contributions.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
