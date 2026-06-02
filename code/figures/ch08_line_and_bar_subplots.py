"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 8 - Data Visualization.

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


def main() -> None:
    """Generate a figure combining a line plot and a bar chart in subplots."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    rng = np.random.default_rng(seed=21)
    n = 50
    x = np.arange(n)
    y_line = np.cumsum(rng.normal(loc=0.0, scale=1.0, size=n))
    y_bar = rng.normal(loc=0.0, scale=1.0, size=n)

    fig, (ax_line, ax_bar) = plt.subplots(
        2,
        1,
        figsize=(7.5, 4.5),
        sharex=True,
    )

    ax_line.plot(x, y_line, color="tab:blue", marker="o", markersize=3)
    ax_line.set_ylabel("Line value")
    ax_line.set_title("Line and Bar Subplots")
    ax_line.grid(True, linestyle="--", alpha=0.3)

    ax_bar.bar(x, y_bar, color="tab:green", width=0.9)
    ax_bar.set_ylabel("Bar value")
    ax_bar.set_xlabel("Index")
    ax_bar.grid(True, axis="y", linestyle="--", alpha=0.3)

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch08_line_and_bar_subplots.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
