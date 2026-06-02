"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 14 - Statistics.

This companion module collects chapter-style helpers for diagnostics,
portfolio statistics, efficient-frontier calculations, and a small
Bayesian update example.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.optimize as sco
import scipy.stats as scs


Array = npt.NDArray[np.float64]


def synthetic_prices() -> pd.DataFrame:
    """Return a compact fallback price table."""

    idx = pd.date_range('2027-01-02', periods=12, freq='B', name='Date')
    return pd.DataFrame(
        {
            'AAPL': [
                190.0, 191.5, 193.0, 192.2, 194.1, 195.0,
                196.3, 197.0, 196.8, 198.2, 199.0, 200.4,
            ],
            'NVDA': [
                620.0, 625.0, 632.0, 629.0, 638.0, 644.0,
                649.0, 655.0, 658.0, 664.0, 670.0, 676.0,
            ],
            'SPY': [
                540.0, 541.0, 542.2, 541.8, 543.0, 544.1,
                545.0, 546.2, 546.5, 547.0, 548.1, 549.3,
            ],
            'GLD': [
                210.0, 209.8, 210.5, 211.2, 210.9, 211.7,
                212.0, 212.4, 212.1, 212.8, 213.0, 213.4,
            ],
        },
        index=idx,
    )


def load_prices(path: str | Path | None = None) -> pd.DataFrame:
    """Load the chapter price data with a synthetic fallback."""

    if path is None:
        path = Path(__file__).resolve().parents[2] / 'data' / 'eod_data.csv'
    path = Path(path)
    if path.exists():
        df = pd.read_csv(
            path,
            parse_dates=['Date'],
            index_col='Date',
        ).dropna(how='all')
        df.index.name = 'Date'
        return df[['AAPL', 'NVDA', 'SPY', 'GLD']].dropna()
    return synthetic_prices()


def gbm_log_levels(
    s0: float = 100.0,
    r: float = 0.02,
    sigma: float = 0.2,
    T: float = 1.0,
    n_paths: int = 250_000,
    seed: int = 2027,
) -> Array:
    """Simulate terminal GBM log levels."""

    rng = np.random.default_rng(seed=seed)
    z = rng.standard_normal(n_paths)
    s_T = s0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    return np.log(s_T)


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns."""

    return np.log(prices / prices.shift(1)).dropna()


def descriptive_statistics(array: Array) -> dict[str, float]:
    """Return a small descriptive-statistics dictionary."""

    sta = scs.describe(array, nan_policy='omit')
    return {
        'size': float(sta.nobs),
        'min': float(sta.minmax[0]),
        'max': float(sta.minmax[1]),
        'mean': float(sta.mean),
        'std': float(np.sqrt(sta.variance)),
        'skew': float(sta.skewness),
        'kurtosis': float(sta.kurtosis),
    }


def portfolio_inputs(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return log returns plus annualised mean and covariance estimates."""

    rets = log_returns(prices)
    mean_rets = rets.mean() * 252
    cov_matrix = rets.cov() * 252
    return rets, mean_rets, cov_matrix


def port_ret(weights: Array, mean_rets: pd.Series) -> float:
    """Compute annualised expected portfolio return."""

    return float(np.sum(mean_rets * weights))


def port_vol(weights: Array, cov_matrix: pd.DataFrame) -> float:
    """Compute annualised portfolio volatility."""

    return float(np.sqrt(weights.T @ cov_matrix @ weights))


def random_portfolios(
    mean_rets: pd.Series,
    cov_matrix: pd.DataFrame,
    n_portfolios: int = 2_500,
    seed: int = 2027,
) -> tuple[Array, Array]:
    """Sample long-only random portfolios."""

    rng = np.random.default_rng(seed=seed)
    noa = len(mean_rets)
    prets = np.empty(n_portfolios)
    pvols = np.empty(n_portfolios)
    for i in range(n_portfolios):
        w = rng.random(noa)
        w /= np.sum(w)
        prets[i] = port_ret(w, mean_rets)
        pvols[i] = port_vol(w, cov_matrix)
    return prets, pvols


def efficient_weights(
    target: float,
    mean_rets: pd.Series,
    cov_matrix: pd.DataFrame,
    initial: Array | None = None,
) -> Array:
    """Return long-only minimum-volatility weights for a target return."""

    noa = len(mean_rets)
    if initial is None:
        initial = np.repeat(1.0 / noa, noa)
    result = sco.minimize(
        lambda w: port_vol(w, cov_matrix),
        initial,
        method='SLSQP',
        bounds=tuple((0.0, 1.0) for _ in range(noa)),
        constraints=(
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': lambda w: port_ret(w, mean_rets) - target},
        ),
    )
    if not result.success:
        raise RuntimeError(
            f'frontier optimization failed for target={target:.6f}'
        )
    return np.asarray(result.x, dtype=float)


def efficient_frontier(
    mean_rets: pd.Series,
    cov_matrix: pd.DataFrame,
    max_return: float,
    n_points: int = 50,
) -> tuple[Array, Array]:
    """Trace a robust efficient frontier starting from the GMV portfolio."""

    noa = len(mean_rets)
    gmv = sco.minimize(
        lambda w: port_vol(w, cov_matrix),
        np.repeat(1.0 / noa, noa),
        method='SLSQP',
        bounds=tuple((0.0, 1.0) for _ in range(noa)),
        constraints=({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},),
    )
    if not gmv.success:
        raise RuntimeError('global minimum-variance optimization failed')

    gmv_weights = np.asarray(gmv.x, dtype=float)
    target_returns = np.linspace(
        port_ret(gmv_weights, mean_rets),
        max_return,
        n_points,
    )

    frontier_weights = []
    initial = gmv_weights
    for target in target_returns:
        initial = efficient_weights(
            float(target),
            mean_rets,
            cov_matrix,
            initial=initial,
        )
        frontier_weights.append(initial)
    frontier_weights = np.asarray(frontier_weights)
    frontier_vols = np.array(
        [port_vol(w, cov_matrix) for w in frontier_weights],
        dtype=float,
    )
    return target_returns, frontier_vols


def bayes_posteriors(priors: Array, likelihood: Array) -> Array:
    """Apply Bayes' rule elementwise for a discrete hypothesis set."""

    evidence = float(np.sum(priors * likelihood))
    return priors * likelihood / evidence


def main() -> None:
    """Run a compact chapter-style statistics demo."""

    print(descriptive_statistics(gbm_log_levels()))
    prices = load_prices()
    rets, mean_rets, cov_matrix = portfolio_inputs(prices)
    print(rets.tail())
    print(mean_rets)
    print(cov_matrix)
    prets, pvols = random_portfolios(mean_rets, cov_matrix)
    target_returns, frontier_vols = efficient_frontier(
        mean_rets,
        cov_matrix,
        max_return=float(prets.max()),
    )
    print(prets[:5], pvols[:5])
    print(target_returns[:5], frontier_vols[:5])
    priors = np.array([0.5, 0.5], dtype=float)
    likelihood = np.array([30 / 90, 60 / 90], dtype=float)
    posteriors = bayes_posteriors(priors, likelihood)
    print(posteriors)
    assert abs(float(posteriors.sum()) - 1.0) < 1e-6, "posteriors must sum to 1"


if __name__ == '__main__':
    main()
