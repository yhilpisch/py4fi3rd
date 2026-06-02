"""Python for Finance, 3rd ed., O'Reilly (2026).
Lab 05 - Why Failed Backtests Still Look Convincing.

Figure: Look-ahead bias creates an impossible equity curve.

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

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.labs.lab05_failed_backtests import equity_curve
from code.labs.lab05_failed_backtests import lookahead_comparison


def main() -> None:
    """Generate the look-ahead bias equity comparison."""
    mpl.use("Agg", force=True)
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

    curves = lookahead_comparison().apply(equity_curve)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for col in curves.columns:
        ax.plot(curves.index, curves[col], label=col)
    ax.set_yscale("log")
    ax.set_title("Look-ahead bias can dominate the whole result")
    ax.set_ylabel("Equity curve, log scale")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    figures_dir = ROOT / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "lab05_lookahead_equity.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
