"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 29 - Derivatives Valuation.

Monte Carlo convergence for LSM valuation of an American put option.

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
import numpy as np

S0 = 36.0
K = 40.0
RATE = 0.06
VOL = 0.2
TTM = 1.0
STEPS = 50
BASIS_DEGREE = 2
SEED = 7

PATH_GRID = np.array([5_000, 10_000, 20_000, 50_000, 100_000], dtype=int)


def main() -> None:
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    code_dir = base_dir / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    from dxlib import AmericanPut, AmericanPutLSM, FlatDiscounting
    from dxlib import GeometricBrownianMotion

    disc = FlatDiscounting(rate=RATE)
    gbm_q = GeometricBrownianMotion(
        drift=RATE,
        volatility=VOL,
        seed=SEED,
    )
    payoff = AmericanPut(strike=K)

    prices: list[float] = []
    errors: list[float] = []
    for paths in PATH_GRID:
        lsm = AmericanPutLSM(
            process=gbm_q,
            payoff=payoff,
            discounting=disc,
            maturity=TTM,
            steps=STEPS,
            paths=int(paths),
            basis_degree=BASIS_DEGREE,
            min_itm=50,
        )
        res = lsm.value(spot=S0)
        prices.append(float(res["price"]))
        errors.append(float(res["stderr"]))

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.errorbar(
        PATH_GRID,
        prices,
        yerr=errors,
        fmt="o-",
        capsize=3,
        linewidth=1.2,
        markersize=4,
    )
    ax.set_xscale("log")
    ax.set_title("LSM Convergence (American Put)")
    ax.set_xlabel("Number of paths (log scale)")
    ax.set_ylabel("Option value")
    ax.grid(True, linestyle="--", alpha=0.25)

    fig.tight_layout()
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch29_mc_convergence.png"
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

