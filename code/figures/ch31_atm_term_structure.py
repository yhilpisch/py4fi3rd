"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 31 - Market-Based Valuation.

ATM implied-vol term structure from the options snapshot.

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

RATE = 0.03
MIN_PRICE = 0.5
MNY_RANGE = (0.85, 1.15)


def main() -> None:
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    code_dir = base_dir / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    from dxlib import implied_vol_forward, load_spx_snapshot, parity_table
    from dxlib import estimate_forward
    from dxlib.time import time_to_maturity

    snap = load_spx_snapshot(base_dir / "data" / "spx_options_snapshot.csv")
    expiries = sorted(snap.raw["EXPIR_DATE"].unique())

    ttms: list[float] = []
    atms: list[float] = []
    for expiry in expiries:
        parity = parity_table(snap, expiry)
        ttm = time_to_maturity(snap.pricing_date, expiry)
        df = float(np.exp(-RATE * ttm))
        fwd = estimate_forward(parity, discount_factor=df)

        parity = parity.copy()
        parity["MNY"] = parity["STRIKE"] / fwd
        parity = parity[parity["CALL_MID"] >= MIN_PRICE]
        parity = parity[
            (parity["MNY"] >= MNY_RANGE[0]) & (parity["MNY"] <= MNY_RANGE[1])
        ]
        if parity.empty:
            continue

        parity["abs_mny"] = (parity["MNY"] - 1.0).abs()
        row = parity.sort_values("abs_mny").iloc[0]
        iv = implied_vol_forward(
            price=float(row["CALL_MID"]),
            forward=float(fwd),
            strike=float(row["STRIKE"]),
            maturity=float(ttm),
            option_type="call",
            discount_factor=df,
        )
        ttms.append(float(ttm))
        atms.append(float(iv))

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.plot(ttms, atms, "o-", linewidth=1.4, markersize=4)
    ax.set_title("ATM Implied Volatility Term Structure")
    ax.set_xlabel("Time to maturity (years)")
    ax.set_ylabel("Implied volatility")
    ax.grid(True, linestyle="--", alpha=0.25)

    fig.tight_layout()
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch31_atm_term_structure.png"
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
