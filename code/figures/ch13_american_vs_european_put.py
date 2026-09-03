"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 13 - Stochastics.

European vs. American (LSM) put option prices across strikes.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from math import erf, log, sqrt, exp


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def euro_put_bs(s0: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0.0:
        return max(K - s0, 0.0)
    d1 = (log(s0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return K * exp(-r * T) * norm_cdf(-d2) - s0 * norm_cdf(-d1)


def lsm_american_put(
    s0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> float:
    """Least-Squares Monte Carlo American put on a GBM underlying."""
    dt = T / n_steps
    shocks = rng.standard_normal((n_steps, n_paths))
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    log_paths = np.vstack([np.zeros(n_paths), log_returns.cumsum(axis=0)])
    S = s0 * np.exp(log_paths)

    h = np.maximum(K - S, 0.0)
    cashflow = h[-1].copy()
    discount = np.exp(-r * dt)

    for t in range(n_steps - 1, 0, -1):
        cashflow *= discount
        in_the_money = h[t] > 0.0
        if not np.any(in_the_money):
            continue
        X = S[t, in_the_money] / K
        Y = cashflow[in_the_money]
        A = np.column_stack([np.ones_like(X), X, X**2])
        coeffs, *_ = np.linalg.lstsq(A, Y, rcond=None)
        continuation = A @ coeffs
        exercise = h[t, in_the_money]
        exercise_now = exercise > continuation
        idx = np.where(in_the_money)[0][exercise_now]
        cashflow[idx] = exercise[exercise_now]

    return float((cashflow * discount).mean())


def main() -> None:
    """Plot European vs. American put prices as a function of strike."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    s0 = 100.0
    r = 0.02
    sigma = 0.2
    T = 1.0
    n_steps = 50
    n_paths = 100_000

    rng = np.random.default_rng(seed=2029)
    strikes = np.linspace(60.0, 140.0, 9)

    euro = np.array([euro_put_bs(s0, K, r, sigma, T) for K in strikes])
    amer = np.array(
        [
            lsm_american_put(s0, K, r, sigma, T, n_steps, n_paths, rng)
            for K in strikes
        ]
    )

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(
        strikes,
        euro,
        marker="o",
        linestyle="-",
        color="tab:blue",
        label="European put",
    )
    ax.plot(
        strikes,
        amer,
        marker="s",
        linestyle="--",
        color="tab:red",
        label="American put (LSM)",
    )
    ax.set_xlabel("Strike")
    ax.set_ylabel("Option price")
    ax.set_title("European vs. American Put Prices (Monte Carlo / LSM)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch13_american_vs_european_put.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
