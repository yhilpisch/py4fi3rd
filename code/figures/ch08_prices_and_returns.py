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
    """Generate a two-panel prices/returns figure and save it as a PNG."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    rng = np.random.default_rng(seed=42)
    n = 200
    steps = rng.normal(loc=0.0005, scale=0.01, size=n)
    prices = 100 * (1 + steps).cumprod()
    rets = steps

    fig, (ax_price, ax_ret) = plt.subplots(
        2,
        1,
        figsize=(7.5, 4.5),
        sharex=True,
    )

    ax_price.plot(prices, color="tab:blue")
    ax_price.set_title("Synthetic Prices and Returns")
    ax_price.set_ylabel("Price")
    ax_price.grid(True, linestyle="--", alpha=0.6)

    ax_ret.bar(np.arange(len(rets)), rets, color="tab:orange", width=0.9)
    ax_ret.axhline(0.0, color="black", linewidth=0.8)
    ax_ret.set_ylabel("Return")
    ax_ret.set_xlabel("Time")
    ax_ret.grid(True, axis="y", linestyle="--", alpha=0.6)

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch08_prices_and_returns.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
