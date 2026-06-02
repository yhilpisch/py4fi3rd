"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 13 - Stochastics.

Sample paths of a discrete-time random walk.

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
    """Plot several random-walk paths."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    rng = np.random.default_rng(seed=123)
    n_steps = 250
    n_paths = 5
    shocks = rng.standard_normal((n_steps, n_paths))
    steps = shocks / np.sqrt(n_steps)
    walk_increments = steps.cumsum(axis=0)
    walk = np.vstack([np.zeros(n_paths), walk_increments])

    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(0, n_steps + 1)
    for j in range(n_paths):
        ax.plot(x, walk[:, j], linewidth=1.0, alpha=0.9)

    ax.set_title("Discrete-Time Random Walk Paths")
    ax.set_xlabel("Step")
    ax.set_ylabel("Level")
    ax.grid(True, linestyle="--", alpha=0.3)

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch13_random_walk_paths.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
