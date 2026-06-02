"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 31 - Market-Based Valuation.

Heston price and price-difference diagnostics for the 2x5 surface.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
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
import pandas as pd

RATE = 0.03
EXPIRIES = [dt.date(2026, 6, 18), dt.date(2026, 12, 18)]
MONEYNESS_TARGETS = [0.8, 0.9, 1.0, 1.05, 1.1]

PATHS = 15_000
STEPS_PER_YEAR = 320
SEED = 11
EVAL_PATHS = 15_000
EVAL_SEED = 23

MARKET_COLOR = "tab:blue"
GLOBAL_COLOR = "tab:green"
LOCAL_COLOR = "tab:red"
BAND_COLOR = "0.65"


def main() -> None:
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {"font.family": "serif", "figure.dpi": 300, "font.size": 9}
    )

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    code_dir = base_dir / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    from dxlib import calibrate_heston_global
    from dxlib import calibrate_heston_local_theta_v0_rho
    from dxlib import evaluate_heston_fit_table
    from dxlib import load_spx_snapshot, select_small_surface
    from dxlib.heston import HestonParams

    snap = load_spx_snapshot(base_dir / "data" / "spx_options_snapshot.csv")
    surface = select_small_surface(
        snap,
        EXPIRIES,
        rate=RATE,
        calibrate_to="CALL",
        moneyness_targets=MONEYNESS_TARGETS,
    )

    global_params, _ = calibrate_heston_global(
        surface,
        rate=RATE,
        paths=PATHS,
        steps_per_year=STEPS_PER_YEAR,
        n_candidates=40,
        n_refine=25,
        seed=SEED,
    )
    local_map, _ = calibrate_heston_local_theta_v0_rho(
        surface,
        global_params=global_params,
        rate=RATE,
        paths=EVAL_PATHS,
        steps_per_year=STEPS_PER_YEAR,
        seed=EVAL_SEED,
    )

    global_table = evaluate_heston_fit_table(
        surface,
        rate=RATE,
        params=global_params,
        paths=EVAL_PATHS,
        steps_per_year=STEPS_PER_YEAR,
        seed=EVAL_SEED,
    )

    local_parts: list[pd.DataFrame] = []
    for expiry in EXPIRIES:
        sub = surface[surface["EXPIR_DATE"] == expiry]
        theta, v0, rho = local_map[expiry]
        params = HestonParams(
            kappa=global_params.kappa,
            theta=theta,
            vol_of_vol=global_params.vol_of_vol,
            rho=rho,
            v0=v0,
        )
        local_parts.append(
            evaluate_heston_fit_table(
                sub,
                rate=RATE,
                params=params,
                paths=EVAL_PATHS,
                steps_per_year=STEPS_PER_YEAR,
                seed=EVAL_SEED,
            )
        )
    local_table = pd.concat(local_parts, axis=0, ignore_index=True)

    n_exp = len(EXPIRIES)
    fig, axes = plt.subplots(
        2, n_exp, figsize=(4.2 * n_exp, 5.4), sharex="col"
    )
    axes_arr = np.asarray(axes)

    for col, expiry in enumerate(EXPIRIES):
        ax_price = axes_arr[0, col]
        ax_diff = axes_arr[1, col]
        mkt = surface[surface["EXPIR_DATE"] == expiry].sort_values("MNY")
        glo = global_table[global_table["EXPIR_DATE"] == expiry]
        loc = local_table[local_table["EXPIR_DATE"] == expiry]
        glo = glo.sort_values("MNY")
        loc = loc.sort_values("MNY")

        mny = mkt["MNY"].to_numpy(dtype=float)
        market_price = mkt["CALL_MID"].to_numpy(dtype=float)
        half_spread = mkt["CALL_HALF_SPREAD"].to_numpy(dtype=float)
        global_price = glo["MODEL_PRICE"].to_numpy(dtype=float)
        local_price = loc["MODEL_PRICE"].to_numpy(dtype=float)

        ax_price.errorbar(
            mny,
            market_price,
            yerr=half_spread,
            fmt="D",
            color=MARKET_COLOR,
            ecolor=MARKET_COLOR,
            elinewidth=0.8,
            capsize=2.0,
            markersize=4.5,
            label="Market call mid",
        )
        ax_price.plot(
            mny,
            global_price,
            "o-",
            color=GLOBAL_COLOR,
            linewidth=1.2,
            markersize=3.5,
            label="Heston (global)",
        )
        ax_price.plot(
            mny,
            local_price,
            "o--",
            color=LOCAL_COLOR,
            linewidth=1.2,
            markersize=3.5,
            label="Heston (local)",
        )
        ax_price.set_title(f"Expiry {expiry.isoformat()}")
        ax_price.set_ylabel("Option price")
        ax_price.grid(True, linestyle="--", alpha=0.25)

        ax_diff.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax_diff.fill_between(
            mny,
            -half_spread,
            half_spread,
            color=BAND_COLOR,
            alpha=0.25,
            label="Bid-ask half-spread",
        )
        ax_diff.plot(
            mny,
            global_price - market_price,
            "o-",
            color=GLOBAL_COLOR,
            linewidth=1.2,
            markersize=3.5,
            label="Heston (global)",
        )
        ax_diff.plot(
            mny,
            local_price - market_price,
            "o--",
            color=LOCAL_COLOR,
            linewidth=1.2,
            markersize=3.5,
            label="Heston (local)",
        )
        ax_diff.set_xlabel("Moneyness K/F")
        ax_diff.set_ylabel("Model - market mid")
        ax_diff.grid(True, linestyle="--", alpha=0.25)

    handles_price, labels_price = axes_arr[0, 0].get_legend_handles_labels()
    handles_diff, labels_diff = axes_arr[1, 0].get_legend_handles_labels()
    handles = handles_price + handles_diff[:1]
    labels = labels_price + labels_diff[:1]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Call prices and price differences", fontsize=10)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.95))

    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch31_heston_calibration_prices.png"
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
