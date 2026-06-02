"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 18 - Portfolio Construction and Risk.

Figure: Factor exposures for a simple two-factor model.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_exposures(universe: list[str]) -> pd.DataFrame:
    """Construct a simple market/rates exposure matrix."""

    exposures = pd.DataFrame(
        {"MKT": [1.0, 1.0, 0.0], "RATES": [0.0, 0.0, 1.0]},
        index=universe,
    )
    return exposures


def main() -> None:
    mpl.use("Agg", force=True)
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

    universe = ["AAPL", "JPM", "TLT"]  # small asset universe
    exposures = build_exposures(universe)  # market and rates exposures

    fig, ax = plt.subplots(figsize=(6, 4))  # create figure and axes

    x = np.arange(len(universe))  # x-locations
    width = 0.35  # bar width

    ax.bar(x - width / 2, exposures["MKT"], width, label="MKT")
    ax.bar(x + width / 2, exposures["RATES"], width, label="RATES")

    ax.set_xticks(x)
    ax.set_xticklabels(universe)
    ax.set_ylabel("Factor exposure")
    ax.set_title("Simple market and rates factor exposures")
    ax.set_ylim(0, 1.2)
    ax.legend(loc="best")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch18_factor_exposures.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
