"""Python for Finance, 3rd ed., O'Reilly (2026).
Lab 07 - Constructing the VIX from SPX Option Data.

Figure: Strike-level contributions to model-free variance.

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

from code.labs.lab07_vix_construction import build_vix_calculation


def main() -> None:
    """Generate the strike-contribution figure for the near maturity."""
    mpl.use("Agg", force=True)
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

    calc = build_vix_calculation()
    term = calc["terms"][0]
    table = term.contribution_table.copy()
    table["scaled"] = table["contribution"] * 1_000_000

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(table["strike"], table["scaled"], width=12.0, color="tab:blue")
    ax.axvline(term.forward, color="black", linestyle="--", label="Forward")
    ax.axvline(term.k0, color="tab:red", linestyle=":", label="K0")
    ax.set_title("Option-strip contributions to variance")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Contribution × 1,000,000")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()

    figures_dir = ROOT / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "lab07_vix_strike_contributions.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
