"""Python for Finance, 3rd ed., O'Reilly (2026).
Lab 06 - Alpha Is Rare.

Figure: Cumulative wealth for the hedge fund index and SPY.

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

from code.labs.lab06_alpha_is_rare import cumulative_wealth
from code.labs.lab06_alpha_is_rare import load_returns


def main() -> None:
    """Generate cumulative wealth paths."""
    mpl.use("Agg", force=True)
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

    returns = load_returns()
    wealth = cumulative_wealth(returns)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(
        wealth.index,
        wealth["HF_INDEX"],
        linewidth=1.8,
        label="Hedge fund index",
    )
    ax.plot(wealth.index, wealth["SPY"], linewidth=1.8, label="SPY")
    ax.set_title("Cumulative wealth: hedge fund index vs SPY")
    ax.set_ylabel("Growth of 1.0")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")

    figures_dir = ROOT / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "lab06_cumulative_wealth.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
