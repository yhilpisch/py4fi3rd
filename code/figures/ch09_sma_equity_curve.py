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
import numpy as np
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
    """Plot equity curves for an SMA crossover strategy vs. buy-and-hold."""

    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

    df = load_eod()
    spy = df["SPY"]
    spy_rets = spy.pct_change().dropna()

    window_short = 21
    window_long = 63

    sma_short = spy.rolling(window_short).mean()
    sma_long = spy.rolling(window_long).mean()
    sma_df = pd.DataFrame(
        {"price": spy, "sma_short": sma_short, "sma_long": sma_long}
    ).dropna()

    position = np.where(
        sma_df["sma_short"] > sma_df["sma_long"], 1.0, 0.0
    )  # long-or-flat
    pos = pd.Series(position, index=sma_df.index)

    aligned_rets = spy_rets.reindex(sma_df.index).fillna(0.0)
    strat_rets = pos.shift(1).fillna(0.0) * aligned_rets
    equity_strategy = (1.0 + strat_rets).cumprod()
    equity_buy_hold = (
        1.0 + spy_rets.reindex(equity_strategy.index).fillna(0.0)
    ).cumprod()

    equity = pd.DataFrame(
        {"strategy": equity_strategy, "buy_and_hold": equity_buy_hold}
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    equity.plot(ax=ax)
    ax.set_title("SMA Strategy vs. Buy-and-Hold (SPY)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (starting at 1.0)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    assets_dir = base_dir / "assets"
    figures_dir = assets_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    outfile = figures_dir / "ch09_sma_equity_curve.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)

    # Write a short textual summary with terminal equity values so that
    # the chapter can include up-to-date numbers via an AsciiDoc include.
    final_strategy = float(equity_strategy.iloc[-1])
    final_bh = float(equity_buy_hold.iloc[-1])
    summary_path = assets_dir / "ch09_sma_equity_summary.txt"
    summary_text = (
        f"As of the end of the sample, the SMA strategy reaches an equity of "
        f"{final_strategy:.3f}, compared to {final_bh:.3f} "
        "for buy-and-hold in SPY.\n"
    )
    summary_path.write_text(summary_text, encoding="utf8")


if __name__ == "__main__":
    main()
