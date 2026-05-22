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
    """Generate a scatter plot of a linear relationship with noise."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    rng = np.random.default_rng(seed=42)
    n = 500
    x = rng.normal(loc=0.0, scale=0.01, size=n)
    noise = rng.normal(loc=0.0, scale=0.002, size=n)
    y = 0.5 * x + noise

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, s=10, alpha=0.6, color="tab:blue", edgecolors="none")
    ax.set_title("Linear Relationship with Noise")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax.grid(True, linestyle="--", alpha=0.3)

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch08_scatter_returns.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
