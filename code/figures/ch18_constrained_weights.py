"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 18 - Portfolio Construction and Risk.

Figure: Unconstrained vs constrained mean–variance portfolio weights.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_returns() -> tuple[pd.DataFrame, list[str]]:
    """Load daily returns for the small Chapter 18 universe."""

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    local = base_dir / "data" / "eod_data.csv"
    remote = "https://hilpisch.com/eod_data.csv"
    source: str | pathlib.Path = local if local.exists() else remote

    prices = pd.read_csv(source, parse_dates=["Date"], index_col="Date")

    universe = ["AAPL", "JPM", "TLT"]
    cols = universe + ["SPY"]
    sub = prices[cols].dropna(how="any")
    rets_all = sub.pct_change().dropna()
    rets = rets_all.iloc[-2 * 252 :]
    return rets, universe


def compute_portfolios(
    rets: pd.DataFrame, universe: list[str]
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute unconstrained and constrained mean–variance portfolios."""

    def project_capped_simplex(w: np.ndarray, cap: float) -> np.ndarray:
        """Project weights onto {w >= 0, sum(w)=1, w_i <= cap}."""

        w = np.asarray(w, dtype=float).copy()
        w = np.clip(w, 0.0, None)
        if cap * len(w) < 1.0:
            raise ValueError("cap too low to allow full investment")

        remaining = np.ones(len(w), dtype=bool)
        out = np.zeros(len(w), dtype=float)
        mass = 1.0

        while True:
            if remaining.sum() == 0:
                return out

            s = w[remaining].sum()
            if s == 0.0:
                out[remaining] = mass / remaining.sum()
                return out

            scaled = w[remaining] * (mass / s)
            over = scaled > cap
            if not np.any(over):
                out[remaining] = scaled
                return out

            idx_over = np.where(remaining)[0][over]
            out[idx_over] = cap
            remaining[idx_over] = False
            mass = 1.0 - out[~remaining].sum()

    mu_daily = rets[universe].mean()
    cov_daily = rets[universe].cov()

    mu_annual = (1 + mu_daily) ** 252 - 1
    cov_annual = cov_daily * 252

    lam = 0.2
    diag_cov = np.diag(np.diag(cov_annual.values))
    cov_shrunk = lam * cov_annual.values + (1 - lam) * diag_cov
    cov_shrunk = pd.DataFrame(cov_shrunk, index=universe, columns=universe)

    Sigma = cov_shrunk.values
    mu_vec = mu_annual.values

    w_mv_uncon = np.linalg.solve(Sigma, mu_vec)
    w_mv_uncon = w_mv_uncon / w_mv_uncon.sum()
    w_mv_uncon = pd.Series(w_mv_uncon, index=universe, name="Unconstrained")

    w_mv_longonly = w_mv_uncon.clip(lower=0)
    w_mv_longonly = w_mv_longonly / w_mv_longonly.sum()
    w_mv_longonly.name = "Long-only"

    cap = 0.35
    w_mv_capped = pd.Series(
        project_capped_simplex(w_mv_longonly.values, cap=cap),
        index=universe,
    )
    w_mv_capped.name = "Long-only, capped"

    return w_mv_uncon, w_mv_longonly, w_mv_capped


def main() -> None:
    mpl.use("Agg", force=True)
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

    rets, universe = load_returns()  # daily returns and asset universe
    w_uncon, w_longonly, w_capped = compute_portfolios(
        rets, universe
    )  # three portfolios

    weights = pd.concat(
        [w_uncon, w_longonly, w_capped], axis=1
    )  # align weights

    fig, ax = plt.subplots(figsize=(6, 4))  # create figure and axes

    x = np.arange(len(universe))  # bar positions
    width = 0.25  # bar width

    ax.bar(x - width, weights["Unconstrained"], width, label="Unconstrained")
    ax.bar(x, weights["Long-only"], width, label="Long-only")
    ax.bar(
        x + width,
        weights["Long-only, capped"],
        width,
        label="Long-only, capped",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(universe)
    ax.set_ylabel("Portfolio weight")
    ax.set_title("Unconstrained vs constrained mean–variance weights")
    ax.legend(loc="best")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch18_constrained_weights.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
