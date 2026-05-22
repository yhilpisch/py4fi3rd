"""Python for Finance, 3rd ed., O'Reilly (2026).
Lab 04 - The Hidden Costs of Portfolio Constraints.

Figure: Trade-off metrics under different constraint sets.

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

from code.labs.lab04_constraint_costs import build_portfolios
from code.labs.lab04_constraint_costs import summarize_tradeoffs


def main() -> None:
    """Generate a four-panel trade-off figure."""
    mpl.use("Agg", force=True)
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

    weights, _, _ = build_portfolios()
    stats = summarize_tradeoffs(weights).set_index("portfolio")

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.4))
    cols = [
        ("exp_return", "Expected return"),
        ("volatility", "Volatility"),
        ("turnover", "Turnover from equal weight"),
        ("tracking_error", "Tracking error vs SPY"),
    ]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for ax, (col, title) in zip(axes.ravel(), cols):
        ax.bar(stats.index, stats[col], color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=12)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    figures_dir = ROOT / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "lab04_constraint_tradeoffs.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
