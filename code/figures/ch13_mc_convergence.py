"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 13 - Stochastics.

Convergence of the Monte Carlo call price estimator.

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
from math import erf, sqrt


def mc_call_price(
    rng: np.random.Generator,
    s0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
) -> float:
    """Estimate a European call price via Monte Carlo."""
    dt = T / n_steps
    shocks = rng.standard_normal((n_steps, n_paths))
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    log_paths = log_returns.cumsum(axis=0)
    s_T = s0 * np.exp(log_paths[-1])
    payoffs = np.maximum(s_T - 100.0, 0.0)
    discount = np.exp(-r * T)
    return float(discount * payoffs.mean())


def main() -> None:
    """Plot Monte Carlo convergence for a call option."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    s0 = 100.0
    K = 100.0
    r = 0.02
    sigma = 0.2
    T = 1.0
    n_steps = 252

    rng = np.random.default_rng(seed=2027)
    grid = np.logspace(3, 6, num=12, dtype=int)
    estimates = [
        mc_call_price(rng, s0, r, sigma, T, n_steps, int(n)) for n in grid
    ]

    # Black–Scholes benchmark value for the same parameters.
    def norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    d1 = (np.log(s0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    bs_price = s0 * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(
        grid,
        estimates,
        marker="o",
        linestyle="-",
        color="tab:blue",
        label="MC estimate",
    )
    ax.axhline(
        bs_price,
        color="tab:red",
        linestyle="--",
        linewidth=1.25,
        label="Black-Scholes price",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Number of paths (log scale)")
    ax.set_ylabel("Call price estimate")
    ax.set_title("Convergence of Monte Carlo Call Price")
    ax.legend(loc="best")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch13_mc_convergence.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
