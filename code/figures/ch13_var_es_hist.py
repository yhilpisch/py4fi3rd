"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 13 - Stochastics.

Histogram of simulated P&L with VaR and Expected Shortfall markers.

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
    """Simulate P&L distribution and show VaR/ES."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    rng = np.random.default_rng(seed=123)
    s0 = 100.0
    r = 0.0
    sigma = 0.25
    T = 30.0 / 365.0
    n_paths = 250_000

    z = rng.standard_normal(n_paths)
    s_T = s0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    pnl = s_T - s0

    losses = -pnl  # positive numbers are losses
    alpha = 0.99
    var_level = np.quantile(losses, alpha)
    es_level = losses[losses >= var_level].mean()

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.hist(losses, bins=80, density=True, color="tab:blue", alpha=0.7)
    ax.axvline(
        var_level,
        color="tab:red",
        linestyle="--",
        linewidth=1.25,
        label=f"VaR {alpha:.0%}",
    )
    ax.axvline(
        es_level,
        color="tab:orange",
        linestyle="-.",
        linewidth=1.25,
        label="Expected Shortfall",
    )

    ax.set_title("Loss Distribution with VaR and Expected Shortfall")
    ax.set_xlabel("Loss")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch13_var_es_hist.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
