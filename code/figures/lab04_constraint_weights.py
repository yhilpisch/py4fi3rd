"""Python for Finance, 3rd ed., O'Reilly (2026).
Lab 04 - The Hidden Costs of Portfolio Constraints.

Figure: Portfolio weights under different constraint sets.

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
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.labs.lab04_constraint_costs import build_portfolios
from code.labs.lab04_constraint_costs import UNIVERSE


def main() -> None:
    """Generate a weight comparison figure for the lab."""
    mpl.use("Agg", force=True)
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

    weights, _, _ = build_portfolios()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    x = np.arange(len(UNIVERSE))
    width = 0.24
    ax.bar(
        x - width,
        weights["Unconstrained"],
        width,
        label="Unconstrained",
    )
    ax.bar(x, weights["Asset cap"], width, label="Asset cap")
    ax.bar(
        x + width,
        weights["Asset + tech cap"],
        width,
        label="Asset + tech cap",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(UNIVERSE)
    ax.set_ylabel("Weight")
    ax.set_title("Weights under different constraint sets")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    figures_dir = ROOT / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "lab04_constraint_weights.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
