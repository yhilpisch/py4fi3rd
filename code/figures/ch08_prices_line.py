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
    """Generate a simple price line chart and save it as a PNG."""
    mpl.style.use("seaborn-v0_8")
    rng = np.random.default_rng(seed=42)
    n = 100
    steps = rng.normal(loc=0.0, scale=0.01, size=n)
    prices = 100 * (1 + steps).cumprod()

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(prices, color="tab:blue", linewidth=1.25)
    ax.set_title("Synthetic Price Series")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.grid(True, linestyle="--", alpha=0.3)

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch08_prices_line.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
