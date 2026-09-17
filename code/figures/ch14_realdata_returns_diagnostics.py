"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 14 - Statistics.

Diagnostics for real-world daily log returns (histograms and QQ plots).

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
from scipy import stats


def main() -> None:
    """Generate histograms and QQ plots for SPY and AAPL log returns."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    data_path = base_dir / "data" / "eod_data.csv"

    url = "https://hilpisch.com/eod_data.csv"

    try:
        data = pd.read_csv(
            data_path, index_col="Date", parse_dates=True
        ).dropna()
    except FileNotFoundError:
        data = pd.read_csv(url, index_col="Date", parse_dates=True).dropna()

    symbols = ["SPY", "AAPL"]
    prices = data[symbols].dropna()

    log_returns = np.log(prices / prices.shift(1)).dropna()

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7))

    for row, sym in enumerate(symbols):
        series = log_returns[sym].to_numpy()
        mu_hat = float(series.mean())
        sigma_hat = float(series.std(ddof=1))

        # Histogram with fitted normal PDF.
        ax = axes[row, 0]
        counts, bins, _ = ax.hist(
            series,
            bins=50,
            density=True,
            color="tab:blue",
            alpha=0.7,
            label=f"{sym} log returns",
        )
        x = np.linspace(bins[0], bins[-1], 300)
        pdf = stats.norm.pdf(x, loc=mu_hat, scale=sigma_hat)
        ax.plot(
            x,
            pdf,
            color="tab:red",
            linewidth=1.5,
            label="Fitted normal PDF",
        )
        ax.set_title(f"{sym}: Histogram and Fitted Normal")
        ax.set_xlabel("log return")
        ax.set_ylabel("Density")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best")

        # QQ plot.
        ax = axes[row, 1]
        stats.probplot(
            series,
            dist="norm",
            sparams=(mu_hat, sigma_hat),
            plot=ax,
        )
        ax.set_title(f"{sym}: QQ Plot vs Normal")
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()

    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch14_realdata_returns_diagnostics.png"
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
