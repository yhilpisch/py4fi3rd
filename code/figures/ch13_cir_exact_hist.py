"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 13 - Stochastics.

Square-root diffusion (CIR) exact discretization: histogram at maturity.

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
    """Generate a histogram for the CIR exact discretization at maturity."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    rng = np.random.default_rng(seed=2028)

    x0 = 0.05
    kappa = 3.0
    theta = 0.02
    sigma = 0.1
    T = 1.0
    n_steps = 252
    n_paths = 200_000
    dt = T / n_steps

    # Parameters for the noncentral chi-square distribution.
    df = 4.0 * theta * kappa / (sigma**2)
    c = sigma**2 * (1.0 - np.exp(-kappa * dt)) / (4.0 * kappa)

    x = np.empty(n_paths)
    x[:] = x0
    for _ in range(n_steps):
        nc = (
            4.0
            * kappa
            * np.exp(-kappa * dt)
            / (sigma**2 * (1.0 - np.exp(-kappa * dt)))
            * x
        )
        x = c * rng.noncentral_chisquare(df=df, nonc=nc, size=n_paths)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.hist(x, bins=80, density=True, color="tab:green", alpha=0.7)
    ax.set_title("CIR Exact Discretization: Values at Maturity")
    ax.set_xlabel("x(T)")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.3)

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch13_cir_exact_hist.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

