"""Python for Finance, 3rd ed., O'Reilly (2026).
Lab 07 - Constructing the VIX from SPX Option Data.

Figure: Interpolation from listed expiries to thirty days.

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

from code.labs.lab07_vix_construction import TARGET_DAYS
from code.labs.lab07_vix_construction import build_vix_calculation


def main() -> None:
    """Generate the total-variance interpolation figure."""
    mpl.use("Agg", force=True)
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

    calc = build_vix_calculation()
    terms = calc["terms"]
    days = [term.days for term in terms]
    total_var = [term.ttm * term.variance for term in terms]
    target_total = calc["variance_30"] * TARGET_DAYS / 365

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(days, total_var, marker="o", label="Listed expiries")
    ax.scatter(
        [TARGET_DAYS],
        [target_total],
        color="tab:red",
        zorder=3,
        label="30-day target",
    )
    ax.set_title("Linear interpolation in total variance")
    ax.set_xlabel("Calendar days to expiry")
    ax.set_ylabel("Total variance")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    figures_dir = ROOT / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "lab07_vix_interpolation.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
