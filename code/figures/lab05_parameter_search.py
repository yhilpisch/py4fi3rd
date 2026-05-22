"""Python for Finance, 3rd ed., O'Reilly (2026).
Lab 05 - Why Failed Backtests Still Look Convincing.

Figure: In-sample parameter winners need not survive out of sample.

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

from code.labs.lab05_failed_backtests import parameter_search


def main() -> None:
    """Generate the parameter-search comparison figure."""
    mpl.use("Agg", force=True)
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

    results = parameter_search()
    xpos = np.arange(len(results))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(
        xpos - width / 2,
        results["train_sharpe"],
        width,
        label="Train",
    )
    ax.bar(
        xpos + width / 2,
        results["test_sharpe"],
        width,
        label="Test",
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(results["lookback"].astype(str))
    ax.set_xlabel("Momentum lookback")
    ax.set_ylabel("Annualized Sharpe ratio")
    ax.set_title("Parameter search can pick unstable winners")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()

    figures_dir = ROOT / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "lab05_parameter_search.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
