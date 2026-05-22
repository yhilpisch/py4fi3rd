"""Python for Finance, 3rd ed., O'Reilly (2026).
Lab 06 - Alpha Is Rare.

Figure: Risk and return trade-offs for hedge fund comparisons.

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

from code.labs.lab06_alpha_is_rare import load_returns
from code.labs.lab06_alpha_is_rare import scenario_summary


def main() -> None:
    """Generate a compact scenario comparison figure."""
    mpl.use("Agg", force=True)
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

    stats = scenario_summary(load_returns()).set_index("portfolio")
    stats = stats.loc[
        ["Hedge fund index", "SPY", "60/40 SPY-IEF", "Levered hedge fund"]
    ]

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.4), sharex=True)
    columns = [
        ("annualized_return", "Annualized return"),
        ("annualized_vol", "Annualized volatility"),
        ("max_drawdown", "Maximum drawdown"),
    ]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    x = np.arange(len(stats.index))
    labels = ["HF", "SPY", "60/40", "Levered HF"]

    for ax, (column, title) in zip(axes, columns):
        values = stats[column].to_numpy()
        ax.bar(x, values, color=colors)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        if column == "max_drawdown":
            ax.axhline(0.0, color="black", linewidth=0.8)

    figures_dir = ROOT / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "lab06_risk_return_tradeoff.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
