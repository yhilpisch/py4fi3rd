"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 22 - Efficient Markets and Hypothesis Testing.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../new
DATA_PATH = PROJECT_ROOT / "data" / "eod_data.csv"
OUT_PATH = PROJECT_ROOT / "assets" / "figures" / "ch22_spy_acf.png"

SYMBOL = "SPY"
MAX_LAG = 30

FIGSIZE = (6.5, 4.0)
DPI = 300
BASE_FONT_SIZE = 10
TITLE_FONT_SIZE = 11


def main() -> None:
    """Plot the autocorrelation of daily returns for a benchmark symbol."""

    mpl.style.use("seaborn-v0_8")  # baseline style
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": BASE_FONT_SIZE,
            "axes.titlesize": TITLE_FONT_SIZE,
        }
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    prices = pd.read_csv(
        DATA_PATH,
        parse_dates=["Date"],
        index_col="Date",
    )
    returns = prices[SYMBOL].pct_change().dropna()  # daily returns

    lags = np.arange(1, MAX_LAG + 1)
    acf_vals = np.array([returns.autocorr(lag=int(k)) for k in lags])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(lags, acf_vals, width=0.8, color="C0", alpha=0.8)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Lag (trading days)")
    ax.set_ylabel("Sample autocorrelation")
    ax.set_title(
        f"Autocorrelation of Daily {SYMBOL} Returns "
        f"(Up to {MAX_LAG} Lags)"
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    main()
