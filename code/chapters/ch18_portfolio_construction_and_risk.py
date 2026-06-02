"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 18 - Portfolio Construction and Risk.

Companion code for key examples in Chapter 18:

- estimation of annualized returns and covariances
- shrinkage of expected returns and covariance matrix
- GMV and constrained mean–variance portfolios
- turnover between portfolios
- asset-level risk contributions
- simple two-factor model and factor risk decomposition

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_EOD = ROOT / "data" / "eod_data.csv"


def load_returns() -> Tuple[pd.DataFrame, list[str]]:
    """Load daily returns for the small Chapter 18 universe and benchmark."""

    if not DATA_EOD.exists():
        msg = f"Expected EOD data at {DATA_EOD}; file not found."
        raise FileNotFoundError(msg)

    prices = pd.read_csv(DATA_EOD, parse_dates=["Date"], index_col="Date")
    universe = ["AAPL", "JPM", "TLT"]
    cols = universe + ["SPY"]
    sub = prices[cols].dropna(how="any")

    rets_all = sub.pct_change().dropna()
    max_rows = 2 * 252
    if len(rets_all) > max_rows:
        rets = rets_all.iloc[-max_rows:]
    else:
        rets = rets_all
    return rets, universe


def estimate_inputs(
    rets: pd.DataFrame,
    universe: list[str],
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Estimate annualized returns, covariances, and benchmark returns."""

    mu_daily = rets[universe].mean()
    cov_daily = rets[universe].cov()

    mu_annual = (1 + mu_daily) ** 252 - 1
    cov_annual = cov_daily * 252
    r_bench = rets["SPY"]
    return mu_annual, cov_annual, r_bench


def shrink_inputs(
    mu_annual: pd.Series,
    cov_annual: pd.DataFrame,
    r_bench: pd.Series,
) -> tuple[pd.Series, pd.DataFrame]:
    """Apply simple shrinkage to expected returns and covariance matrix."""

    mu_bench_annual = (1 + r_bench.mean()) ** 252 - 1
    shrink = 0.5
    mu_shrunk = shrink * mu_annual + (1 - shrink) * mu_bench_annual

    lam = 0.2
    diag_cov = np.diag(np.diag(cov_annual.values))
    cov_shrunk = lam * cov_annual.values + (1 - lam) * diag_cov
    cov_shrunk_df = pd.DataFrame(
        cov_shrunk,
        index=cov_annual.index,
        columns=cov_annual.columns,
    )
    return mu_shrunk, cov_shrunk_df


def gmv_weights(cov_shrunk: pd.DataFrame) -> np.ndarray:
    """Compute global minimum-variance (GMV) portfolio weights."""

    Sigma = cov_shrunk.values
    ones = np.ones(Sigma.shape[0])
    inv_Sigma_ones = np.linalg.solve(Sigma, ones)
    w_gmv = inv_Sigma_ones / (ones @ inv_Sigma_ones)
    return w_gmv


def mean_variance_portfolios(
    mu_shrunk: pd.Series,
    cov_shrunk: pd.DataFrame,
    universe: list[str],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute unconstrained and constrained mean–variance portfolios."""

    Sigma = cov_shrunk.values
    mu_vec = mu_shrunk.values

    w_mv_uncon = np.linalg.solve(Sigma, mu_vec)
    w_mv_uncon = w_mv_uncon / w_mv_uncon.sum()
    w_mv_uncon_s = pd.Series(
        w_mv_uncon,
        index=universe,
        name="Unconstrained",
    )

    w_mv_longonly = w_mv_uncon_s.clip(lower=0)
    w_mv_longonly = w_mv_longonly / w_mv_longonly.sum()
    w_mv_longonly.name = "Long-only"

    cap = 0.35
    w_mv_capped = w_mv_longonly.clip(upper=cap)
    w_mv_capped = w_mv_capped / w_mv_capped.sum()
    w_mv_capped.name = "Long-only, capped"
    return w_mv_uncon_s, w_mv_longonly, w_mv_capped


def turnover(w_from: np.ndarray, w_to: np.ndarray) -> float:
    """Compute turnover as half the sum of absolute weight differences."""

    w_from = np.asarray(w_from, dtype=float)
    w_to = np.asarray(w_to, dtype=float)
    return 0.5 * np.abs(w_to - w_from).sum()


def risk_contributions(
    weights: np.ndarray,
    cov: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    """Compute absolute risk contributions and total variance."""

    w = np.asarray(weights, dtype=float)
    Sigma = np.asarray(cov, dtype=float)
    marginal = Sigma @ w
    total_var = float(w @ marginal)
    contrib = w * marginal
    return contrib, total_var


def build_factor_model(
    rets: pd.DataFrame,
    universe: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Construct a simple two-factor (MKT, RATES) model."""

    factors = ["MKT", "RATES"]
    exposures = pd.DataFrame(
        {"MKT": [1.0, 1.0, 0.0], "RATES": [0.0, 0.0, 1.0]},
        index=universe,
    )

    F = pd.DataFrame(
        {"MKT": rets["SPY"], "RATES": rets["TLT"]},
        index=rets.index,
    )
    cov_f_annual = F.cov() * 252

    B = exposures.values
    cov_factor_implied = B @ cov_f_annual.values @ B.T
    cov_factor_implied_df = pd.DataFrame(
        cov_factor_implied,
        index=universe,
        columns=universe,
    )
    return exposures, cov_f_annual, cov_factor_implied_df, factors


def factor_risk_contributions(
    w_port: np.ndarray,
    exposures: pd.DataFrame,
    cov_f_annual: pd.DataFrame,
    factors: list[str],
) -> pd.Series:
    """Compute factor-level percentage risk contributions for a portfolio."""

    w = np.asarray(w_port, dtype=float)
    b_port = exposures.T @ w
    cov_f = cov_f_annual.values
    marginal_f = cov_f @ b_port
    var_f = float(b_port @ marginal_f)
    rc_f = b_port * marginal_f
    rc_f_pct = pd.Series(
        rc_f / var_f,
        index=factors,
        name="factor_rc_pct",
    )
    return rc_f_pct


def main() -> None:
    """Run a compact Chapter 18 demo and print key results."""

    rets, universe = load_returns()
    mu_annual, cov_annual, r_bench = estimate_inputs(rets, universe)
    mu_shrunk, cov_shrunk = shrink_inputs(mu_annual, cov_annual, r_bench)

    print("== Shrunk expected returns (annualised) ==")
    print(mu_shrunk)
    print()

    w_gmv = gmv_weights(cov_shrunk)
    print("== GMV weights ==")
    print(w_gmv)
    print()

    w_mv_uncon, w_mv_longonly, w_mv_capped = mean_variance_portfolios(
        mu_shrunk,
        cov_shrunk,
        universe,
    )
    print("== Unconstrained mean–variance weights ==")
    print(w_mv_uncon)
    print()

    print("== Long-only, capped mean–variance weights ==")
    print(w_mv_capped)
    print()

    w_eq = np.repeat(1.0 / len(universe), len(universe))
    to_mv = turnover(w_eq, w_mv_capped.values)
    print("== Turnover from equal-weight to constrained MV ==")
    print(f"{to_mv:.6f}")
    print()

    rc_mv, var_mv = risk_contributions(w_mv_capped.values, cov_shrunk)
    rc_mv_pct = pd.Series(rc_mv / var_mv, index=universe)
    print("== Asset-level percentage risk contributions ==")
    print(rc_mv_pct)
    print()

    exposures, cov_f_annual, cov_factor_implied, factors = build_factor_model(
        rets,
        universe,
    )
    print("== Factor exposure matrix ==")
    print(exposures)
    print()

    print("== Factor-implied asset covariance matrix ==")
    print(cov_factor_implied)
    print()

    rc_f_pct = factor_risk_contributions(
        w_mv_capped.values,
        exposures,
        cov_f_annual,
        factors,
    )
    print("== Factor-level percentage risk contributions ==")
    print(rc_f_pct)

    assert abs(float(w_mv_capped.sum()) - 1.0) < 1e-6, "MV weights must sum to 1"
    assert abs(float(rc_mv_pct.sum()) - 1.0) < 1e-6, (
        "asset risk contributions must sum to 1"
    )


if __name__ == "__main__":
    main()
