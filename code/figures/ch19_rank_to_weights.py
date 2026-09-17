"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 19 - Signals, Forecasts, and Portfolio Implementation.

Figure: Standardized signals and corresponding long-only portfolio weights.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_returns() -> tuple[pd.DataFrame, list[str]]:
    """Load daily returns for a small universe and benchmark."""

    base_dir = Path(__file__).resolve().parents[2]
    local = base_dir / "data" / "eod_data.csv"
    remote = "https://hilpisch.com/eod_data.csv"
    source: str | Path = local if local.exists() else remote

    prices = pd.read_csv(source, parse_dates=["Date"], index_col="Date")

    universe = ["AAPL", "JPM", "TLT"]
    cols = universe + ["SPY"]
    sub = prices[cols].dropna(how="any")

    rets = sub[universe].pct_change().dropna()
    rets = rets.iloc[-2 * 252 :]
    return rets, universe


def compute_latest_z_and_weights(
    rets: pd.DataFrame,
    universe: list[str],
) -> tuple[pd.Series, pd.Series]:
    """Compute latest momentum z-scores and corresponding weights."""

    mom_20d = rets.rolling(20).mean()
    latest = mom_20d.iloc[-1]
    z_scores = (latest - latest.mean()) / latest.std()
    z_pos = z_scores.clip(lower=0)
    if z_pos.sum() == 0:
        weights = pd.Series(
            np.repeat(1.0 / len(universe), len(universe)),
            index=universe,
        )
    else:
        weights = (z_pos / z_pos.sum()).reindex(universe)
    return z_scores.reindex(universe), weights


def main() -> None:
    mpl.use("Agg", force=True)
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif"})
    mpl.rcParams.update({"figure.dpi": 300})

    rets, universe = load_returns()
    z_scores, weights = compute_latest_z_and_weights(rets, universe)

    x = np.arange(len(universe))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(9.5, 4.0),
        sharey=False,
    )

    ax1.bar(x, z_scores.values, width, color="tab:blue")
    ax1.axhline(0.0, color="black", linewidth=0.8, alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(universe)
    ax1.set_ylabel("Momentum z-score")
    ax1.set_title("Standardized signals")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.3)

    ax2.bar(x, weights.values, width, color="tab:green")
    ax2.set_xticks(x)
    ax2.set_xticklabels(universe)
    ax2.set_ylabel("Portfolio weight")
    ax2.set_title("Long-only weights from positive z-scores")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()

    base_dir = Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch19_rank_to_weights.png"
    fig.savefig(outfile)
    plt.close(fig)


if __name__ == "__main__":
    main()
