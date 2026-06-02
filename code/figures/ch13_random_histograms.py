"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 13 - Stochastics.

Histograms of uniform and standard normal random samples.

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
    """Generate histograms for uniform and normal samples."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    rng = np.random.default_rng(seed=42)
    u = rng.random(100_000)
    z = rng.standard_normal(100_000)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharey=False)

    ax = axes[0]
    ax.hist(u, bins=40, density=True, color="tab:blue", alpha=0.7)
    ax.set_title("Uniform Samples on [0, 1)")
    ax.set_xlabel("u")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.3)

    ax = axes[1]
    ax.hist(z, bins=40, density=True, color="tab:orange", alpha=0.7)
    ax.set_title("Standard Normal Samples")
    ax.set_xlabel("z")
    ax.grid(True, linestyle="--", alpha=0.3)

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch13_random_histograms.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

