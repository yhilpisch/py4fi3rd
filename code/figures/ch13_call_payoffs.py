"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 13 - Stochastics.

Distribution of discounted Monte Carlo call payoffs.

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
    """Simulate discounted call payoffs and show their distribution."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    rng = np.random.default_rng(seed=1234)

    s0 = 100.0
    r = 0.02
    sigma = 0.2
    T = 1.0
    K = 100.0
    n_steps = 252
    n_paths = 250_000
    dt = T / n_steps

    shocks = rng.standard_normal((n_steps, n_paths))
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    log_paths = log_returns.cumsum(axis=0)
    s_T = s0 * np.exp(log_paths[-1])
    payoffs = np.maximum(s_T - K, 0.0)
    discount = np.exp(-r * T)
    discounted = discount * payoffs

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.hist(discounted, bins=60, density=True, color="tab:blue", alpha=0.7)
    ax.set_title("Discounted Monte Carlo Call Payoffs")
    ax.set_xlabel("Discounted payoff")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.3)

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch13_call_payoffs.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

