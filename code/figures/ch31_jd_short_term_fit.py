"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 31 - Market-Based Valuation.

Short-horizon smile fit: jump diffusion versus Heston on one expiry.

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

RATE = 0.03
EXPIRY = dt.date(2026, 3, 20)
MONEYNESS_TARGETS = [0.8, 0.9, 1.0, 1.05, 1.1]

PATHS = 15_000
STEPS_PER_YEAR = 320
SEED = 11
EVAL_SEED = 23


def main() -> None:
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

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

    fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.2))
    mny = mkt["MNY"].to_numpy(dtype=float)
    jitter = 0.002 * (
        np.arange(mny.size, dtype=float) - 0.5 * float(mny.size - 1)
    )
    ax.plot(
        mny + jitter,
        mkt["IMPL_VOL"],
        "D",
        markersize=5,
        label="Market (calls)",
    )
    ax.plot(
        hes["MNY"],
        hes["MODEL_IV"],
        "o-",
        linewidth=1.2,
        markersize=3.5,
        label="Heston",
    )
    ax.plot(
        jd["MNY"],
        jd["MODEL_IV"],
        "o--",
        linewidth=1.2,
        markersize=3.5,
        label="Jump diffusion",
    )
    ax.set_title(f"Expiry {EXPIRY.isoformat()}")
    ax.set_xlabel("Moneyness K/F")
    ax.set_ylabel("Implied volatility")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()

    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch31_jd_short_term_fit.png"
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
