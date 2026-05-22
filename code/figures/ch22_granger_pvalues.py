"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 22 - Efficient Markets and Hypothesis Testing.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from pathlib import Path
import contextlib  # suppress stdout from statsmodels helper
import io  # in-memory text buffer

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../new
DATA_PATH = PROJECT_ROOT / "data" / "eod_data.csv"
OUT_PATH = PROJECT_ROOT / "assets" / "figures" / "ch22_granger_pvalues.png"

SYMBOL = "SPY"
WINDOW = 20
MAX_LAG = 5
P_THRESHOLD = 0.05

FIGSIZE = (6.5, 4.0)
DPI = 300
BASE_FONT_SIZE = 10
TITLE_FONT_SIZE = 11


def main() -> None:
    """Plot Granger-causality p-values for a simple momentum signal."""

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

    momentum = returns.rolling(WINDOW).mean()  # simple momentum

    data_gc = pd.concat(
        {"r": returns, f"mom_{WINDOW}d": momentum},
        axis=1,
    ).dropna()

    buf = io.StringIO()  # buffer for printed output
    with contextlib.redirect_stdout(buf):  # hide test summaries
        tests = grangercausalitytests(
            data_gc,
            maxlag=MAX_LAG,
        )

    lags = np.arange(1, MAX_LAG + 1)
    pvalues = np.array(
        [tests[lag][0]["ssr_ftest"][1] for lag in lags],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(lags, pvalues, marker="o", color="C1", label="p-value")
    ax.axhline(
        P_THRESHOLD,
        color="grey",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
        label=f"{int(P_THRESHOLD * 100)}% level",
    )
    ax.set_xlabel("Lag in Granger-causality test")
    ax.set_ylabel("p-value")
    ax.set_title(f"Granger p-values: momentum vs next-day {SYMBOL}")
    ax.set_xticks(lags)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=DPI)


if __name__ == "__main__":
    main()
