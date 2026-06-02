"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 28 - Simulation of Financial Models.

Terminal distribution comparison for GBM vs jump diffusion vs Heston.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
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
import numpy as np


def main() -> None:
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    code_dir = base_dir / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    from dxlib import (
        GeometricBrownianMotion,
        HestonModel,
        JumpDiffusion,
        build_time_grid,
    )

    seed = 2829
    spot = 100.0
    maturity = 1.0
    steps = 252
    paths = 50_000
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

    gbm_terminal = gbm.simulate_paths(
        spot=spot,
        time_grid=grid,
        paths=paths,
    )[:, -1]
    jd_terminal = jd.simulate_paths(
        spot=spot,
        time_grid=grid,
        paths=paths,
    )[:, -1]
    hes_spot, _ = heston.simulate_paths(
        spot=spot,
        variance=0.04,
        time_grid=grid,
        paths=paths,
    )
    hes_terminal = hes_spot[:, -1]

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    bins = np.linspace(40, 200, 80)
    ax.hist(gbm_terminal, bins=bins, density=True, alpha=0.45, label="GBM")
    ax.hist(
        jd_terminal,
        bins=bins,
        density=True,
        alpha=0.45,
        label="Jump diffusion",
    )
    ax.hist(hes_terminal, bins=bins, density=True, alpha=0.45, label="Heston")
    ax.set_title("Terminal Price Distributions (1Y)")
    ax.set_xlabel("Terminal price")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="upper right")

    fig.tight_layout()
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch28_terminal_distributions.png"
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
