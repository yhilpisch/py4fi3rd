"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 13 - Stochastics.

Square-root diffusion (CIR) simulated via Euler scheme: histogram at maturity
and sample paths.

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


def simulate_cir_euler(
    x0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate CIR paths with an Euler scheme."""
    dt = T / n_steps
    x = np.empty((n_steps + 1, n_paths), dtype=float)
    x[0] = x0
    for t in range(1, n_steps + 1):
        z = rng.standard_normal(n_paths)
        x_prev = x[t - 1]
        x[t] = x_prev + kappa * (theta - x_prev) * dt + sigma * np.sqrt(
            np.maximum(x_prev, 0.0)
        ) * np.sqrt(dt) * z
        x[t] = np.maximum(x[t], 0.0)
    return x


def main() -> None:
    """Generate CIR Euler histogram at maturity and a few sample paths."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    rng = np.random.default_rng(seed=2027)
    x0 = 0.05
    kappa = 3.0
    theta = 0.02
    sigma = 0.1
    T = 1.0
    n_steps = 252
    n_paths_hist = 200_000
    n_paths_paths = 10

    cir_all = simulate_cir_euler(
        x0=x0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths_hist,
        rng=rng,
    )

    # Histogram at maturity.
    x_T = cir_all[-1]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))

    ax = axes[0]
    ax.hist(x_T, bins=80, density=True, color="tab:blue", alpha=0.7)
    ax.set_title("CIR Euler Scheme: Values at Maturity")
    ax.set_xlabel("x(T)")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.3)

    # Sample paths.
    cir_paths = simulate_cir_euler(
        x0=x0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths_paths,
        rng=rng,
    )

    ax = axes[1]
    x_axis = np.linspace(0.0, T, n_steps + 1)
    for j in range(n_paths_paths):
        ax.plot(x_axis, cir_paths[:, j], linewidth=1.0, alpha=0.9)
    ax.set_title("CIR Euler Scheme: Sample Paths")
    ax.set_xlabel("Time")
    ax.set_ylabel("x(t)")
    ax.grid(True, linestyle="--", alpha=0.3)

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch13_cir_euler_hist_and_paths.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

