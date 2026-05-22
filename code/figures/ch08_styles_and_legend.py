"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 8 - Data Visualization.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """Generate a line plot with two series, styles, and a legend."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    rng = np.random.default_rng(seed=123)
    n = 100
    steps = rng.normal(loc=0.0, scale=0.02, size=(n, 2))
    data = steps.cumsum(axis=0)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(
        data[:, 0],
        color="tab:blue",
        linestyle="-",
        linewidth=1.4,
        marker="o",
        markersize=3,
        label="Series A",
    )
    ax.plot(
        data[:, 1],
        color="tab:orange",
        linestyle="--",
        linewidth=1.4,
        marker="s",
        markersize=3,
        label="Series B",
    )
    ax.set_title("Two Series with Styles and Legend")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch08_styles_and_legend.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
