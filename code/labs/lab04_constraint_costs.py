"""Python for Finance, 3rd ed., O'Reilly (2026).
Lab 04 - The Hidden Costs of Portfolio Constraints.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "eod_data.csv"
UNIVERSE = ["AAPL", "NVDA", "JPM", "SPY"]
BENCHMARK = "SPY"
TECH = ["AAPL", "NVDA"]


def load_returns(window: int = 2 * 252) -> tuple[pd.DataFrame, pd.Series]:
    """Load daily returns for the lab universe and benchmark."""
    prices = pd.read_csv(DATA_FILE, parse_dates=["Date"], index_col="Date")
    cols = list(dict.fromkeys(UNIVERSE + [BENCHMARK]))
    sub = prices[cols].dropna(how="any")
    rets = sub.pct_change().dropna()
    if len(rets) > window:
        rets = rets.iloc[-window:]
    return rets[UNIVERSE], rets[BENCHMARK]


def estimate_moments(
    rets: pd.DataFrame, shrink: float = 0.25
) -> tuple[pd.Series, pd.DataFrame]:
    """Estimate annualized means and a shrunk covariance matrix."""
    mu_daily = rets.mean()
    cov_daily = rets.cov()

    mu_annual = (1.0 + mu_daily) ** 252 - 1.0
    cov_annual = cov_daily * 252.0

    diag_cov = np.diag(np.diag(cov_annual.values))
    shrunk = shrink * cov_annual.values + (1.0 - shrink) * diag_cov
    Sigma = pd.DataFrame(shrunk, index=UNIVERSE, columns=UNIVERSE)
    return mu_annual, Sigma


def project_capped_simplex(w: np.ndarray, cap: float) -> np.ndarray:
    """Project onto long-only, fully invested weights with a max cap."""
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
        idx = np.where(remaining)[0][over]
        out[idx] = cap
        remaining[idx] = False
        mass = 1.0 - out[~remaining].sum()


def enforce_group_cap(
    weights: pd.Series, group: list[str], cap: float
) -> pd.Series:
    """Cap the total group weight and redistribute the excess."""
    out = weights.copy()
    group_weight = float(out[group].sum())
    if group_weight <= cap:
        return out

    excess = group_weight - cap
    out[group] *= cap / group_weight

    other = [name for name in out.index if name not in group]
    other_weight = float(out[other].sum())
    if other_weight == 0.0:
        out[other] = excess / len(other)
    else:
        out[other] += excess * out[other] / other_weight
    return out / out.sum()


def build_portfolios() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Construct three portfolios and return market data for analysis."""
    asset_rets, bench = load_returns()
    mu, Sigma = estimate_moments(asset_rets)

    w_uncon = np.linalg.solve(Sigma.values, mu.values)
    w_uncon = w_uncon / w_uncon.sum()
    uncon = pd.Series(w_uncon, index=UNIVERSE, name="Unconstrained")

    long_only = uncon.clip(lower=0.0)
    long_only = long_only / long_only.sum()
    capped = pd.Series(
        project_capped_simplex(long_only.values, cap=0.35),
        index=UNIVERSE,
        name="Asset cap",
    )

    sector_capped = enforce_group_cap(capped, TECH, cap=0.25)
    sector_capped.name = "Asset + tech cap"

    weights = pd.concat([uncon, capped, sector_capped], axis=1)
    return weights, asset_rets, bench


def turnover(w_from: np.ndarray, w_to: np.ndarray) -> float:
    """Compute one-way turnover between two weight vectors."""
    return 0.5 * np.abs(np.asarray(w_to) - np.asarray(w_from)).sum()


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Compute fractional variance contributions."""
    w = weights.values
    marginal = cov.values @ w
    total = float(w @ marginal)
    contrib = w * marginal / total
    return pd.Series(contrib, index=weights.index)


def summarize_tradeoffs(weights: pd.DataFrame) -> pd.DataFrame:
    """Summarize return, volatility, turnover, and tracking error."""
    asset_rets, bench = load_returns()
    mu, Sigma = estimate_moments(asset_rets)
    eq = np.repeat(1.0 / len(UNIVERSE), len(UNIVERSE))

    rows = []
    for name in weights.columns:
        w = weights[name]
        port = asset_rets @ w.values
        active = port - bench
        rows.append(
            {
                "portfolio": name,
                "exp_return": float(w @ mu),
                "volatility": float(np.sqrt(w @ Sigma @ w)),
                "turnover": turnover(eq, w.values),
                "tracking_error": float(active.std() * np.sqrt(252.0)),
                "herfindahl": float((w**2).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Print a compact summary of portfolio trade-offs."""
    weights, asset_rets, _ = build_portfolios()
    _, Sigma = estimate_moments(asset_rets)
    tradeoffs = summarize_tradeoffs(weights)
    rc = risk_contributions(weights["Asset + tech cap"], Sigma)
    print(weights.round(3).to_string())
    print()
    print(tradeoffs.round(3).to_string(index=False))
    print()
    print(rc.round(3).to_string())


if __name__ == "__main__":
    main()
