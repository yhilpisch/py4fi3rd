"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 28 - Simulation of Financial Models.

Comparison of simulated price paths under GBM, jump diffusion, and Heston.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import os
import pathlib
import sys

_MPLCONFIGDIR = pathlib.Path(
    os.environ.get("MPLCONFIGDIR", "/tmp/mplconfig")
).expanduser()
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_MPLCONFIGDIR)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt


def main() -> None:
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    code_dir = base_dir / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    from dxlib import (
        CIRShortRate,
        GeometricBrownianMotion,
        HestonModel,
        JumpDiffusion,
        build_time_grid,
    )

    seed = 2828
    spot = 100.0
    maturity = 1.0
    steps = 252
    paths = 12
    grid = build_time_grid(maturity, steps)

    gbm = GeometricBrownianMotion(drift=0.03, volatility=0.20, seed=seed)
    jd = JumpDiffusion(
        drift=0.03,
        volatility=0.18,
        jump_intensity=0.6,
        jump_mean=-0.08,
        jump_std=0.25,
        seed=seed,
    )
    heston = HestonModel(
        kappa=2.0,
        theta=0.04,
        vol_of_vol=0.5,
        rho=-0.7,
        drift=0.03,
        seed=seed,
    )
    cir = CIRShortRate(kappa=1.2, theta=0.03, sigma=0.15, seed=seed)

    gbm_paths = gbm.simulate_paths(spot=spot, time_grid=grid, paths=paths)
    jd_paths = jd.simulate_paths(spot=spot, time_grid=grid, paths=paths)
    hes_paths, _ = heston.simulate_paths(
        spot=spot,
        variance=0.04,
        time_grid=grid,
        paths=paths,
    )
    rates = cir.simulate_paths(rate0=0.03, time_grid=grid, paths=paths)

    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.2), sharex=True)
    axes = axes.reshape(2, 2)
    x = grid

    axes[0, 0].plot(x, gbm_paths.T, linewidth=1.0, alpha=0.85)
    axes[0, 0].set_title("GBM (log-normal diffusion)")
    axes[0, 0].set_ylabel("Price")

    axes[0, 1].plot(x, jd_paths.T, linewidth=1.0, alpha=0.85)
    axes[0, 1].set_title("Jump Diffusion (Merton)")

    axes[1, 0].plot(x, hes_paths.T, linewidth=1.0, alpha=0.85)
    axes[1, 0].set_title("Heston (stochastic volatility)")
    axes[1, 0].set_xlabel("Time (years)")
    axes[1, 0].set_ylabel("Price")

    axes[1, 1].plot(x, rates.T, linewidth=1.0, alpha=0.85)
    axes[1, 1].set_title("CIR short rate (paths)")
    axes[1, 1].set_xlabel("Time (years)")
    axes[1, 1].set_ylabel("Rate")

    for ax in axes.ravel():
        ax.grid(True, linestyle="--", alpha=0.25)

    fig.suptitle("Simulated Paths: Model Comparison", y=0.98)
    fig.tight_layout()

    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch28_simulated_paths_comparison.png"
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
