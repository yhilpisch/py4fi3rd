"""Python for Finance, 3rd ed., O'Reilly (2026).
Lab 04 - The Hidden Costs of Portfolio Constraints.

Figure: Risk contributions for constrained portfolios.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
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
from code.labs.lab04_constraint_costs import estimate_moments
from code.labs.lab04_constraint_costs import risk_contributions


def main() -> None:
    """Generate a risk-contribution comparison figure."""
    mpl.use("Agg", force=True)
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

    weights, asset_rets, _ = build_portfolios()
    _, Sigma = estimate_moments(asset_rets)

    left = risk_contributions(weights["Asset cap"], Sigma)
    right = risk_contributions(weights["Asset + tech cap"], Sigma)
    data = np.column_stack([left.values, right.values])

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(left.index))
    width = 0.34
    ax.bar(x - width / 2, data[:, 0], width, label="Asset cap")
    ax.bar(x + width / 2, data[:, 1], width, label="Asset + tech cap")
    ax.set_xticks(x)
    ax.set_xticklabels(left.index)
    ax.set_ylabel("Fraction of variance")
    ax.set_title("Risk contributions under different constraints")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    figures_dir = ROOT / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "lab04_risk_contributions.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
