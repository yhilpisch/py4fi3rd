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
    """Generate a figure with two y-axes for differently scaled series."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    rng = np.random.default_rng(seed=7)
    n = 100
    x = np.arange(n)
    y1 = np.cumsum(rng.normal(loc=0.0, scale=1.0, size=n))
    y2 = 0.01 * np.cumsum(rng.normal(loc=0.0, scale=1.0, size=n))

    fig, ax1 = plt.subplots(figsize=(7.5, 4))
    ax2 = ax1.twinx()

    ax1.plot(x, y1, color="tab:blue", label="Series A (large scale)")
    ax1.set_title("Two Series with Different Scales and Two y-Axes")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Series A", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2.plot(
        x,
        y2,
        color="tab:orange",
        linestyle="--",
        label="Series B (small scale)",
    )
    ax2.set_ylabel("Series B", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.grid(True, linestyle="--", alpha=0.6)

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch08_dual_axes.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
