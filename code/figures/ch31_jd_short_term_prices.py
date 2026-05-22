"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 31 - Market-Based Valuation.

Short-horizon price and price-difference fit diagnostics.

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
EXPIRY = dt.date(2026, 3, 20)
MONEYNESS_TARGETS = [0.8, 0.9, 1.0, 1.05, 1.1]

PATHS = 15_000
STEPS_PER_YEAR = 320
SEED = 11
EVAL_SEED = 23

MARKET_COLOR = "tab:blue"
HESTON_COLOR = "tab:green"
JUMP_COLOR = "tab:red"
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
    from dxlib import calibrate_jump_diffusion_single_expiry
    from dxlib import evaluate_heston_fit_table
    from dxlib import evaluate_jump_diffusion_fit_table
    from dxlib import load_spx_snapshot, select_small_surface

    snap = load_spx_snapshot(base_dir / "data" / "spx_options_snapshot.csv")
    surface = select_small_surface(
        snap,
        [EXPIRY],
        rate=RATE,
        calibrate_to="CALL",
        moneyness_targets=MONEYNESS_TARGETS,
    )

    jd_params, _ = calibrate_jump_diffusion_single_expiry(
        surface,
        rate=RATE,
        paths=PATHS,
        steps_per_year=STEPS_PER_YEAR,
        n_candidates=40,
        n_refine=25,
        seed=SEED,
    )
    jd_table = evaluate_jump_diffusion_fit_table(
        surface,
        rate=RATE,
        params=jd_params,
        paths=PATHS,
        steps_per_year=STEPS_PER_YEAR,
        seed=EVAL_SEED,
    )

    heston_params, _ = calibrate_heston_global(
        surface,
        rate=RATE,
        paths=PATHS,
        steps_per_year=STEPS_PER_YEAR,
        n_candidates=40,
        n_refine=25,
        seed=SEED,
    )
    heston_table = evaluate_heston_fit_table(
        surface,
        rate=RATE,
        params=heston_params,
        paths=PATHS,
        steps_per_year=STEPS_PER_YEAR,
        seed=EVAL_SEED,
    )

    mkt = surface.sort_values("MNY")
    jd = jd_table.sort_values("MNY")
    hes = heston_table.sort_values("MNY")

    mny = mkt["MNY"].to_numpy(dtype=float)
    market_price = mkt["CALL_MID"].to_numpy(dtype=float)
    half_spread = mkt["CALL_HALF_SPREAD"].to_numpy(dtype=float)
    jd_price = jd["MODEL_PRICE"].to_numpy(dtype=float)
    hes_price = hes["MODEL_PRICE"].to_numpy(dtype=float)

    fig, (ax_price, ax_diff) = plt.subplots(
        1, 2, figsize=(8.2, 3.2), sharex=True
    )

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
        hes_price,
        "o-",
        color=HESTON_COLOR,
        linewidth=1.2,
        markersize=3.5,
        label="Heston",
    )
    ax_price.plot(
        mny,
        jd_price,
        "o--",
        color=JUMP_COLOR,
        linewidth=1.2,
        markersize=3.5,
        label="Jump diffusion",
    )
    ax_price.set_title("Call prices")
    ax_price.set_xlabel("Moneyness K/F")
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
        hes_price - market_price,
        "o-",
        color=HESTON_COLOR,
        linewidth=1.2,
        markersize=3.5,
        label="Heston",
    )
    ax_diff.plot(
        mny,
        jd_price - market_price,
        "o--",
        color=JUMP_COLOR,
        linewidth=1.2,
        markersize=3.5,
        label="Jump diffusion",
    )
    ax_diff.set_title("Model - market mid")
    ax_diff.set_xlabel("Moneyness K/F")
    ax_diff.set_ylabel("Price difference")
    ax_diff.grid(True, linestyle="--", alpha=0.25)

    handles_price, labels_price = ax_price.get_legend_handles_labels()
    handles_diff, labels_diff = ax_diff.get_legend_handles_labels()
    handles = handles_price + handles_diff[:1]
    labels = labels_price + labels_diff[:1]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(f"Expiry {EXPIRY.isoformat()}", fontsize=10)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.95))

    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch31_jd_short_term_prices.png"
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
