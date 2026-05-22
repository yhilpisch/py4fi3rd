"""Python for Finance, 3rd ed., O'Reilly (2026).
Lab 05 - Why Failed Backtests Still Look Convincing.

Figure: Trading costs turn a cleaner gross backtest into a weaker net one.

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

from code.labs.lab05_failed_backtests import equity_curve
from code.labs.lab05_failed_backtests import momentum_returns


def main() -> None:
    """Generate the gross-versus-net momentum equity comparison."""
    mpl.use("Agg", force=True)
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

    data = momentum_returns(lookback=20)
    curves = data[["gross", "net"]].apply(equity_curve)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(curves.index, curves["gross"], label="Gross")
    ax.plot(curves.index, curves["net"], label="Net of costs")
    ax.set_title("Transaction costs lower the realized path")
    ax.set_ylabel("Equity curve")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    figures_dir = ROOT / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "lab05_cost_drag.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
