"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 14 - Statistics.

Normality diagnostics for simulated GBM log returns.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import math
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def main() -> None:
    """Generate histogram + QQ plot for simulated GBM log returns."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    rng = np.random.default_rng(seed=2027)
    r = 0.02
    sigma = 0.2
    T = 1.0
    n_steps = 50
    n_paths = 250_000
    dt = T / n_steps

    shocks = rng.standard_normal((n_steps, n_paths))
    # Moment matching for variance reduction (as in the 2nd edition).
    shocks = (shocks - shocks.mean(axis=0, keepdims=True)) / shocks.std(
        axis=0, ddof=0, keepdims=True
    )

    log_returns = (r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * shocks
    log_returns_flat = log_returns.ravel()

    mu_step = log_returns_flat.mean()
    sigma_step = log_returns_flat.std(ddof=1)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))

    # Histogram with fitted normal PDF.
    ax = axes[0]
    counts, bins, _ = ax.hist(
        log_returns_flat,
        bins=70,
        density=True,
        color="tab:blue",
        alpha=0.7,
        label="Simulated log returns",
    )
    x = np.linspace(bins[0], bins[-1], 300)
    pdf = stats.norm.pdf(x, loc=mu_step, scale=sigma_step)
    ax.plot(x, pdf, color="tab:red", linewidth=1.5, label="Fitted normal PDF")
    ax.set_title("Histogram of Simulated GBM Log Returns")
    ax.set_xlabel("log return")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")

    # QQ plot against fitted normal.
    ax = axes[1]
    stats.probplot(
        log_returns_flat,
        dist="norm",
        sparams=(mu_step, sigma_step),
        plot=ax,
    )
    ax.set_title("QQ Plot for Simulated GBM Log Returns")
    ax.grid(True, linestyle="--", alpha=0.3)

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch14_gbm_log_returns_diagnostics.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
