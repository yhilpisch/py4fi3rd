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
import numpy as np
import pandas as pd


def load_prices() -> pd.DataFrame:
    """Load end-of-day prices from the local CSV or remote fallback."""

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    local = base_dir / "data" / "eod_data.csv"
    remote = "https://hilpisch.com/eod_data.csv"
    source: str | pathlib.Path = local if local.exists() else remote
    df = pd.read_csv(source, parse_dates=["Date"], index_col="Date")
    return df


def main() -> None:
    """Illustrate tracking error for an equal-weight portfolio vs SPY."""

    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

    prices = load_prices()

    symbols = ["AAPL", "JPM", "TLT"]
    sub = prices[symbols + ["SPY"]].dropna(how="any")

    rets_all = sub.pct_change().dropna()

    # Focus on a recent window (approximately the last two years of data).
    max_rows = 2 * 252
    if len(rets_all) > max_rows:
        rets = rets_all.iloc[-max_rows:]
    else:
        rets = rets_all

    w_eq = np.repeat(1.0 / len(symbols), len(symbols))
    r_port = (rets[symbols] * w_eq).sum(axis=1)
    r_bench = rets["SPY"]
    active = r_port - r_bench

    te_annual = active.std(ddof=1) * np.sqrt(252.0)

    cum_port = (1.0 + r_port).cumprod()
    cum_bench = (1.0 + r_bench).cumprod()

    fig, (ax_top, ax_bottom) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(7.5, 5.5),
        sharex=True,
    )

    ax_top.plot(cum_port.index, cum_port, label="Equal-weight portfolio")
    ax_top.plot(cum_bench.index, cum_bench, label="SPY benchmark")
    ax_top.set_ylabel("Cumulative value")
    ax_top.set_title("Portfolio vs SPY: cumulative returns")
    ax_top.grid(True, linestyle="--", alpha=0.3)
    ax_top.legend(loc="upper left")

    ax_bottom.plot(
        active.index,
        active,
        label="Active return (portfolio - SPY)",
    )
    ax_bottom.axhline(0.0, color="black", linewidth=0.8, alpha=0.8)
    ax_bottom.set_ylabel("Active return (daily)")
    ax_bottom.set_xlabel("Date")
    ax_bottom.grid(True, linestyle="--", alpha=0.3)
    ax_bottom.legend(
        loc="upper left",
        title=f"Annualised TE ≈ {te_annual:.1%}",
    )

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch17_tracking_error_illustration.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
