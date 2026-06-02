"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 5 - Numerical Computing with NumPy.

This script-style module demonstrates more idiomatic, typed helpers
for Chapter 5 topics:

- vectorized return calculations,
- simple broadcasting-based portfolio operations,
- and a small GBM simulation function with invariants.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


def simple_returns(
    prices: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Compute simple returns from a 1D array of prices."""

    prices = np.asarray(prices, dtype=float)
    return prices[1:] / prices[:-1] - 1.0


def portfolio_returns(
    weights: npt.NDArray[np.floating],
    asset_returns: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Compute portfolio returns for multiple periods via broadcasting.

    ``weights`` is assumed to have shape ``(n_assets,)`` and
    ``asset_returns`` shape ``(n_periods, n_assets)``.
    """

    w = np.asarray(weights, dtype=float)
    r = np.asarray(asset_returns, dtype=float)
    assert r.shape[1] == w.shape[0]
    return r @ w


@dataclass
class GBMParams:
    """Parameters for a simple geometric Brownian motion."""

    s0: float
    mu: float
    sigma: float
    t: float


def simulate_gbm(
    params: GBMParams,
    n_paths: int = 10_000,
    seed: int | None = None,
) -> npt.NDArray[np.floating]:
    """Return terminal prices from a GBM simulation."""

    rng = np.random.default_rng(seed=seed)
    z = rng.standard_normal(size=n_paths)
    s0, mu, sigma, t = params.s0, params.mu, params.sigma, params.t
    st = s0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * z)
    return st


def main() -> None:
    """Run a few self-checks on the helpers."""

    prices = np.array([100.0, 101.5, 103.0, 102.0])
    rets = simple_returns(prices)
    assert rets.shape == (3,)

    asset_r = np.array(
        [
            [0.01, -0.005],
            [0.002, 0.003],
        ]
    )
    w = np.array([0.6, 0.4])
    p_rets = portfolio_returns(w, asset_r)
    assert p_rets.shape == (2,)

    params = GBMParams(s0=100.0, mu=0.05, sigma=0.2, t=1.0)
    st = simulate_gbm(params, n_paths=20_000, seed=123)
    expected_mean = params.s0 * np.exp(params.mu * params.t)
    assert np.isclose(st.mean(), expected_mean, rtol=0.02)


if __name__ == "__main__":
    main()
