"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 9 - Financial Time Series.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
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
    """Plot SPY with short and long simple moving averages."""

    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

    df = load_eod()
    spy = df["SPY"]

    window_short = 21
    window_long = 63

    sma_short = spy.rolling(window_short).mean()
    sma_long = spy.rolling(window_long).mean()

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    spy.plot(ax=ax, color="tab:gray", linewidth=1.0, label="SPY")
    sma_short.plot(ax=ax, color="tab:blue", linewidth=1.4, label="21-day SMA")
    sma_long.plot(ax=ax, color="tab:orange", linewidth=1.4, label="63-day SMA")

    ax.set_title("SPY with Short and Long SMAs")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price level")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch09_sma_crossover.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

