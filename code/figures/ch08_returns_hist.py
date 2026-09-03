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
    """Generate a histogram of synthetic returns and save it as a PNG."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    rng = np.random.default_rng(seed=42)
    n = 500
    rets = rng.normal(loc=0.0005, scale=0.01, size=n)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.hist(rets, bins=30, color="tab:gray", edgecolor="black", alpha=0.8)
    ax.set_title("Synthetic Daily Returns Histogram")
    ax.set_xlabel("Return")
    ax.set_ylabel("Frequency")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch08_returns_hist.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
