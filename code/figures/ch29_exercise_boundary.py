"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 29 - Derivatives Valuation.

Estimated early-exercise boundary from LSM for an American put option.

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
PATHS = 200_000
SEED = 7


def _forward_fill_nan(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    valid = np.isfinite(out)
    if not np.any(valid):
        return out
    last = out[valid][0]
    for idx, value in enumerate(out):
        if np.isfinite(value):
            last = value
        else:
            out[idx] = last
    return out


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
    lsm = AmericanPutLSM(
        process=gbm_q,
        payoff=payoff,
        discounting=disc,
        maturity=TTM,
        steps=STEPS,
        paths=PATHS,
        basis_degree=2,
        min_itm=200,
    )
    res = lsm.value(spot=S0)
    time_grid = np.asarray(res["time_grid"], dtype=float)
    boundary = np.asarray(res["exercise_boundary"], dtype=float)

    boundary = _forward_fill_nan(boundary)

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.plot(time_grid, boundary, linewidth=1.6)
    ax.set_title("Estimated Early-Exercise Boundary (American Put)")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Exercise boundary level")
    ax.grid(True, linestyle="--", alpha=0.25)

    fig.tight_layout()
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch29_exercise_boundary.png"
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

