"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 30 - Portfolio Valuation.

Portfolio delta contributions by position.

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
RATE = 0.06
VOL = 0.2
TTM = 1.0
STEPS = 50
PATHS = 200_000
SEED = 7
REL_BUMP = 0.01


def main() -> None:
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    code_dir = base_dir / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    from dxlib import AmericanPut, AmericanPutLSM
    from dxlib import EuropeanCall, EuropeanMCPricer, EuropeanPut
    from dxlib import FlatDiscounting, GeometricBrownianMotion
    from dxlib import Portfolio, Position

    disc = FlatDiscounting(rate=RATE)
    gbm_q = GeometricBrownianMotion(drift=RATE, volatility=VOL, seed=SEED)

    euro_put_pricer = EuropeanMCPricer(
        process=gbm_q,
        payoff=EuropeanPut(strike=40.0),
        discounting=disc,
        maturity=TTM,
        steps=STEPS,
        paths=PATHS,
    )
    euro_call_pricer = EuropeanMCPricer(
        process=gbm_q,
        payoff=EuropeanCall(strike=40.0),
        discounting=disc,
        maturity=TTM,
        steps=STEPS,
        paths=PATHS,
    )
    am_put_pricer = AmericanPutLSM(
        process=gbm_q,
        payoff=AmericanPut(strike=40.0),
        discounting=disc,
        maturity=TTM,
        steps=STEPS,
        paths=PATHS,
        basis_degree=2,
    )

    positions = (
        Position("American put (K=40)", 1.0, am_put_pricer),
        Position("Short European put (K=40)", -2.0, euro_put_pricer),
        Position("European call (K=40)", 1.0, euro_call_pricer),
    )
    book = Portfolio(name="Index options book", positions=positions)
    deltas = book.deltas(spot=S0, rel_bump=REL_BUMP)
    names = [p.name for p in positions]
    values = np.array([deltas[n] for n in names], dtype=float)

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    colors = ["C0" if v >= 0 else "C3" for v in values]
    ax.bar(names, values, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Portfolio Delta Contributions")
    ax.set_ylabel("Delta")
    ax.tick_params(axis="x", labelrotation=15)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)

    fig.tight_layout()
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch30_portfolio_delta_contributions.png"
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
