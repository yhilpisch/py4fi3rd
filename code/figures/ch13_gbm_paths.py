"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 13 - Stochastics.

Simulated geometric Brownian motion price paths.

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
    """Simulate and plot several GBM paths."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    rng = np.random.default_rng(seed=2027)
    s0 = 100.0
    mu = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = 252
    n_paths = 20
    dt = T / n_steps

    shocks = rng.standard_normal((n_steps, n_paths))
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    log_paths = np.vstack([np.zeros(n_paths), log_returns.cumsum(axis=0)])
    s_paths = s0 * np.exp(log_paths)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(0, n_steps + 1)
    ax.plot(x, s_paths, linewidth=0.6, alpha=0.7, color="tab:blue")
    ax.set_title("Geometric Brownian Motion Paths")
    ax.set_xlabel("Step")
    ax.set_ylabel("Price")
    ax.grid(True, linestyle="--", alpha=0.3)

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch13_gbm_paths.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

