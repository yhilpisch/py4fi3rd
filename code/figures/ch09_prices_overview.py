"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 9 - Financial Time Series.

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


def load_eod() -> pd.DataFrame:
    """Load end-of-day data from the local CSV or remote fallback."""

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    local = base_dir / "data" / "eod_data.csv"
    remote = "https://hilpisch.com/eod_data.csv"
    source: str | pathlib.Path = local if local.exists() else remote
    df = pd.read_csv(source, parse_dates=["Date"], index_col="Date")
    return df


def main() -> None:
    """Generate an overview plot of selected EOD prices using pandas.plot()."""

    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

    df = load_eod()
    subset = df[["AAPL", "SPY", "GLD", "TLT"]]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    subset.plot(ax=ax)
    ax.set_title("Selected End-of-Day Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price level")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch09_prices_overview.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

