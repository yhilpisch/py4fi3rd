"""Python for Finance, 3rd ed., O'Reilly (2026).
Lab 01 - The Importance of Return Tails for Investing and Trading.

Figure: Rolling share of absolute move explained by the five largest days.

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

from code.labs.lab01_return_tails import load_returns
from code.labs.lab01_return_tails import rolling_tail_share


def main() -> None:
    """Generate the rolling tail-share figure."""
    mpl.use("Agg", force=True)
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

    rets = load_returns("SPY")
    share = rolling_tail_share(rets, window=252)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(share.index, share.values, color="tab:purple", linewidth=1.4)
    ax.set_title("Share of absolute annual move explained by five days")
    ax.set_ylabel("Share of absolute move")
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, linestyle="--", alpha=0.3)

    figures_dir = ROOT / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "lab01_tail_share.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
