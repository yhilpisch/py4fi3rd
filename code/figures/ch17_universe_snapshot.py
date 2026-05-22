"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 17 - Asset Management Foundations.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


def load_example_holdings() -> pd.DataFrame:
    """Create a small illustrative holdings table for a simple universe."""

    data = {
        "symbol": ["AAPL", "NVDA", "JPM", "SPY"],
        "quantity": [120, 80, 150, 200],
        "price": [180.25, 820.10, 145.30, 520.10],
        "sector": ["Technology", "Technology", "Financials", "Equity Index"],
        "region": ["US", "US", "US", "Global"],
        "currency": ["USD", "USD", "USD", "USD"],
    }
    holdings = pd.DataFrame(data)
    holdings["market_value"] = holdings["quantity"] * holdings["price"]
    total_market_value = holdings["market_value"].sum()
    holdings["weight"] = holdings["market_value"] / total_market_value
    return holdings


def main() -> None:
    """Generate a bar chart of illustrative portfolio weights."""

    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

    holdings = load_example_holdings()

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar(holdings["symbol"], holdings["weight"])
    ax.set_ylabel("Portfolio weight")
    ax.set_title("Illustrative Cross-Sectional Portfolio Weights")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    for x, w in zip(holdings["symbol"], holdings["weight"]):
        ax.text(
            x,
            w,
            f"{w:.1%}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch17_universe_snapshot.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
