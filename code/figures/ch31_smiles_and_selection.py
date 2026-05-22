"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 31 - Market-Based Valuation.

Implied-vol smiles for three expiries with highlighted 3x5 book surface.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import datetime as dt
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
EXPIRIES = [dt.date(2026, 3, 20), dt.date(2026, 6, 18), dt.date(2026, 12, 18)]
MNY_RANGE = (0.75, 1.25)
MONEYNESS_TARGETS = [0.8, 0.9, 1.0, 1.05, 1.1]
MIN_PRICE = 0.5
SHORT_TTM_THRESHOLD = 0.20
SHORT_MIN_MID_PRICE = 1.0


def main() -> None:
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    code_dir = base_dir / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    from dxlib import implied_vol_forward, load_spx_snapshot, parity_table
    from dxlib import estimate_forward, select_small_surface
    from dxlib.time import time_to_maturity

    snap = load_spx_snapshot(base_dir / "data" / "spx_options_snapshot.csv")
    small = select_small_surface(
        snap,
        EXPIRIES,
        rate=RATE,
        calibrate_to="CALL",
        moneyness_targets=MONEYNESS_TARGETS,
        short_ttm_threshold=SHORT_TTM_THRESHOLD,
        short_min_mid_price=SHORT_MIN_MID_PRICE,
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2), sharey=True)
    for ax, expiry in zip(axes, EXPIRIES, strict=True):
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

        mny = parity["MNY"].to_numpy(dtype=float)
        strike = parity["STRIKE"].to_numpy(dtype=float)
        price = parity["CALL_MID"].to_numpy(dtype=float)

        vols = []
        for k, p in zip(strike, price, strict=True):
            vols.append(
                implied_vol_forward(
                    price=float(p),
                    forward=float(fwd),
                    strike=float(k),
                    maturity=float(ttm),
                    option_type="call",
                    discount_factor=df,
                )
            )
        vols_arr = np.array(vols, dtype=float)

        ax.scatter(mny, vols_arr, s=10, alpha=0.6, label="Calls (snapshot)")

        small_e = small[small["EXPIR_DATE"] == expiry]
        mny_sel = small_e["MNY"].to_numpy(dtype=float)
        jitter = 0.002 * (
            np.arange(mny_sel.size, dtype=float)
            - 0.5 * float(mny_sel.size - 1)
        )
        ax.scatter(
            mny_sel + jitter,
            small_e["IMPL_VOL"],
            s=44,
            marker="D",
            edgecolor="black",
            linewidth=0.7,
            label="Selected (5 strikes)",
        )

        ax.set_title(f"Expiry {expiry.isoformat()}")
        ax.set_xlabel("Moneyness K/F")
        ax.grid(True, linestyle="--", alpha=0.25)

    axes[0].set_ylabel("Implied volatility")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))

    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch31_smiles_and_selection.png"
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
