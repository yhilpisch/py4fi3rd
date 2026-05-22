"""Python for Finance, 3rd ed., O'Reilly (2026).
Lab 01 - The Importance of Return Tails for Investing and Trading.

Figure: Cumulative paths with and without the most extreme days.

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
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.labs.lab01_return_tails import cumulative_path
from code.labs.lab01_return_tails import load_returns
from code.labs.lab01_return_tails import remove_extreme_days


def main() -> None:
    """Generate cumulative paths for full, no-best, and no-worst returns."""
    mpl.use("Agg", force=True)
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

    rets = load_returns("SPY")
    full = cumulative_path(rets)
    no_best = cumulative_path(remove_extreme_days(rets, n=10, side="best"))
    no_worst = cumulative_path(remove_extreme_days(rets, n=10, side="worst"))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(full.index, full.values, label="Full sample", linewidth=1.5)
    ax.plot(no_best.index, no_best.values, label="Without 10 best days")
    ax.plot(no_worst.index, no_worst.values, label="Without 10 worst days")
    ax.set_title("Cumulative growth and the role of extreme days")
    ax.set_ylabel("Cumulative value (start = 1.0)")
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    figures_dir = ROOT / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "lab01_tail_paths.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
