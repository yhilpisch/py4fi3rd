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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (required for 3D)


def main() -> None:
    """Generate a 3D scatter plot and save it as a PNG."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    rng = np.random.default_rng(seed=123)
    n = 400
    x = rng.uniform(0.5, 2.0, size=n)
    y = rng.uniform(0.1, 1.0, size=n)
    z = np.exp(-0.5 * (x + y)) + 0.05 * rng.standard_normal(size=n)

    fig = plt.figure(figsize=(7.5, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, c=z, cmap="viridis", s=15, alpha=0.8)
    ax.set_title("3D Scatter Plot")
    ax.set_xlabel("Maturity")
    ax.set_ylabel("Strike")
    ax.set_zlabel("Value")

    # Place the colorbar in its own axes to leave more room
    cax = fig.add_axes([0.82, 0.2, 0.03, 0.6])
    fig.colorbar(sc, cax=cax)

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch08_scatter_3d.png"
    fig.subplots_adjust(left=0.05, right=0.8, bottom=0.28, top=0.9)
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
