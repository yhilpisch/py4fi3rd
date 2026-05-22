"""Python for Finance, 3rd ed., O'Reilly (2026).
Lab 01 - The Importance of Return Tails for Investing and Trading.

Figure: Histogram of daily returns with tail cutoffs highlighted.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import os
import pathlib
import sys

os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"
os.environ["MPLBACKEND"] = "Agg"

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.labs.lab01_return_tails import load_returns


def main() -> None:
    """Generate the return histogram and mark empirical tail cutoffs."""
    mpl.use("Agg", force=True)
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

    rets = load_returns("SPY")
    q05 = rets.quantile(0.05)
    q95 = rets.quantile(0.95)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(rets, bins=40, color="tab:gray", edgecolor="black", alpha=0.8)
    ax.axvline(q05, color="tab:red", linestyle="--", label="5% tail cutoff")
    ax.axvline(q95, color="tab:green", linestyle="--", label="95% tail cutoff")
    ax.set_title("SPY daily return distribution with tail cutoffs")
    ax.set_xlabel("Simple daily return")
    ax.set_ylabel("Frequency")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    figures_dir = ROOT / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "lab01_tail_hist.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
