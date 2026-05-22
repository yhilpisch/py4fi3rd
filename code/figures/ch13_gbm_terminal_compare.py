"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 13 - Stochastics.

Terminal GBM levels simulated via standard normal and lognormal sampling.

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
    """Compare GBM terminal levels from two simulation approaches."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    rng = np.random.default_rng(seed=2027)

    s0 = 100.0
    r = 0.02
    sigma = 0.2
    T = 1.0
    n_paths = 250_000

    # Standard-normal based simulation from the GBM log formula.
    z = rng.standard_normal(n_paths)
    s_T_norm = s0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)

    # Direct lognormal sampling with matching parameters.
    mean = np.log(s0) + (r - 0.5 * sigma**2) * T
    std = sigma * np.sqrt(T)
    s_T_logn = rng.lognormal(mean=mean, sigma=std, size=n_paths)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    bins = 80
    ax.hist(
        s_T_norm,
        bins=bins,
        density=True,
        alpha=0.6,
        color="tab:blue",
        label="Standard-normal sampling",
    )
    ax.hist(
        s_T_logn,
        bins=bins,
        density=True,
        alpha=0.4,
        color="tab:orange",
        label="Lognormal sampling",
    )

    ax.set_title("Terminal GBM Levels from Two Simulation Schemes")
    ax.set_xlabel("Terminal price $S_T$")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch13_gbm_terminal_compare.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

