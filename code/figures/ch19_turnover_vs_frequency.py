"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 19 - Signals, Forecasts, and Portfolio Implementation.

Figure: Annualized turnover under weekly and monthly rebalancing.

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


def weights_from_signals(signals: pd.DataFrame) -> pd.DataFrame:
    """Map momentum signals to long-only weights."""

    def row_to_weights(row: pd.Series) -> np.ndarray:
        z = (row - row.mean()) / row.std()
        z_pos = z.clip(lower=0)
        if z_pos.sum() == 0:
            return np.repeat(1.0 / len(row), len(row))
        return (z_pos / z_pos.sum()).values

    weights = signals.apply(
        row_to_weights,
        axis=1,
        result_type="expand",
    )
    return weights


def turnover_from_weights(weights: pd.DataFrame) -> float:
    """Compute annualized turnover from a weights table."""

    diff = weights.diff().abs().sum(axis=1)
    daily_turnover = 0.5 * diff
    return float(daily_turnover.mean() * 252.0)


def main() -> None:
    mpl.use("Agg", force=True)
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif"})
    mpl.rcParams.update({"figure.dpi": 300})

    rets, universe = load_returns()
    mom_20d = rets.rolling(20).mean()

    weekly = mom_20d.resample("W-FRI").last().dropna()
    monthly = mom_20d.resample("ME").last().dropna()

    w_weekly = weights_from_signals(weekly)
    w_weekly.columns = universe
    w_weekly = w_weekly.reindex(rets.index, method="ffill")

    w_monthly = weights_from_signals(monthly)
    w_monthly.columns = universe
    w_monthly = w_monthly.reindex(rets.index, method="ffill")

    to_weekly = turnover_from_weights(w_weekly)
    to_monthly = turnover_from_weights(w_monthly)

    freqs = ["Weekly", "Monthly"]
    values = [to_weekly, to_monthly]

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    x = np.arange(len(freqs))

    ax.bar(x, values, width=0.6, color=["tab:blue", "tab:orange"])
    ax.set_xticks(x)
    ax.set_xticklabels(freqs)
    ax.set_ylabel("Annualized turnover")
    ax.set_title("Turnover under different rebalancing frequencies")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    for xpos, val in zip(x, values):
        ax.text(
            xpos,
            val,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()

    base_dir = Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch19_turnover_vs_frequency.png"
    fig.savefig(outfile)
    plt.close(fig)


if __name__ == "__main__":
    main()
