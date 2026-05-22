"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 13 - Stochastics.

This companion module provides small reusable building blocks for the chapter's
simulation workflows: random sampling, stochastic paths, Monte Carlo valuation,
and simple tail-risk metrics.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from math import erf, exp, sqrt

import numpy as np
import numpy.typing as npt


Array = npt.NDArray[np.float64]


def sample_uniform_normal(
    seed: int = 42,
    n: int = 100_000,
) -> tuple[Array, Array]:
    """Draw large uniform and standard normal samples."""

    rng = np.random.default_rng(seed=seed)
    return rng.random(n), rng.standard_normal(n)


def random_walk(seed: int = 123, n_steps: int = 250, n_paths: int = 5) -> Array:
    """Simulate discrete-time random-walk paths."""

    rng = np.random.default_rng(seed=seed)
    shocks = rng.standard_normal((n_steps, n_paths))
    steps = shocks / np.sqrt(n_steps)
    return np.vstack([np.zeros(n_paths), steps.cumsum(axis=0)])


def gbm_paths(
    s0: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.2,
    T: float = 1.0,
    n_steps: int = 252,
    n_paths: int = 20,
    seed: int = 2027,
) -> Array:
    """Simulate geometric Brownian motion price paths."""

    rng = np.random.default_rng(seed=seed)
    dt = T / n_steps
    shocks = rng.standard_normal((n_steps, n_paths))
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    log_paths = np.vstack([np.zeros(n_paths), log_returns.cumsum(axis=0)])
    return s0 * np.exp(log_paths)


def gbm_terminal_levels(
    s0: float = 100.0,
    r: float = 0.02,
    sigma: float = 0.2,
    T: float = 1.0,
    n_paths: int = 250_000,
    seed: int = 2027,
) -> tuple[Array, Array]:
    """Simulate terminal GBM levels via normal and lognormal constructions."""

    rng = np.random.default_rng(seed=seed)
    z = rng.standard_normal(n_paths)
    s_T_norm = s0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    mean = np.log(s0) + (r - 0.5 * sigma**2) * T
    std = sigma * np.sqrt(T)
    s_T_logn = rng.lognormal(mean=mean, sigma=std, size=n_paths)
    return s_T_norm, s_T_logn


def cir_euler(
    x0: float = 0.05,
    kappa: float = 3.0,
    theta: float = 0.02,
    sigma: float = 0.1,
    T: float = 1.0,
    n_steps: int = 252,
    n_paths: int = 50_000,
    seed: int = 2027,
) -> Array:
    """Simulate CIR paths with a truncated Euler scheme."""

    rng = np.random.default_rng(seed=seed)
    dt = T / n_steps
    x = np.empty((n_steps + 1, n_paths), dtype=float)
    x[0] = x0
    for t in range(1, n_steps + 1):
        z = rng.standard_normal(n_paths)
        x_prev = x[t - 1]
        x[t] = x_prev + kappa * (theta - x_prev) * dt + sigma * np.sqrt(
            np.maximum(x_prev, 0.0)
        ) * np.sqrt(dt) * z
        x[t] = np.maximum(x[t], 0.0)
    return x


def cir_exact(
    x0: float = 0.05,
    kappa: float = 3.0,
    theta: float = 0.02,
    sigma: float = 0.1,
    T: float = 1.0,
    n_steps: int = 252,
    n_paths: int = 50_000,
    seed: int = 2028,
) -> Array:
    """Simulate terminal CIR values via exact chi-square transitions."""

    rng = np.random.default_rng(seed=seed)
    dt = T / n_steps
    df = 4.0 * theta * kappa / sigma**2
    c = sigma**2 * (1.0 - np.exp(-kappa * dt)) / (4.0 * kappa)
    x = np.full(n_paths, x0, dtype=float)
    for _ in range(n_steps):
        nc = (
            4.0
            * kappa
            * np.exp(-kappa * dt)
            * x
            / (sigma**2 * (1.0 - np.exp(-kappa * dt)))
        )
        x = c * rng.noncentral_chisquare(df=df, nonc=nc, size=n_paths)
    return x


def mc_european_call(
    s0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int = 252,
    n_paths: int = 250_000,
    seed: int = 2027,
) -> float:
    """Estimate a European call price by Monte Carlo under GBM."""

    rng = np.random.default_rng(seed=seed)
    dt = T / n_steps
    shocks = rng.standard_normal((n_steps, n_paths))
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * sqrt(dt) * shocks
    s_T = s0 * np.exp(log_returns.cumsum(axis=0)[-1])
    return float(exp(-r * T) * np.maximum(s_T - K, 0.0).mean())


def norm_cdf(x: float) -> float:
    """Return the standard normal CDF."""

    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def euro_put_bs(s0: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black-Scholes European put value."""

    if T <= 0.0:
        return max(K - s0, 0.0)
    d1 = (np.log(s0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm_cdf(-d2) - s0 * norm_cdf(-d1))


def lsm_american_put(
    s0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = 2027,
) -> float:
    """Least-Squares Monte Carlo estimate of an American put."""

    dt = T / n_steps
    rng = np.random.default_rng(seed=seed)
    shocks = rng.standard_normal((n_steps, n_paths))
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    log_paths = np.vstack([np.zeros(n_paths), log_returns.cumsum(axis=0)])
    S = s0 * np.exp(log_paths)
    h = np.maximum(K - S, 0.0)
    V = h.copy()

    for t in range(n_steps - 1, 0, -1):
        in_the_money = h[t] > 0.0
        if not np.any(in_the_money):
            continue
        X = S[t, in_the_money]
        Y = V[t + 1, in_the_money] * np.exp(-r * dt)
        A = np.column_stack([np.ones_like(X), X, X**2])
        coeffs, *_ = np.linalg.lstsq(A, Y, rcond=None)
        continuation = A @ coeffs
        exercise = h[t, in_the_money]
        exercise_now = exercise > continuation
        idx = np.where(in_the_money)[0][exercise_now]
        V[t, idx] = exercise[exercise_now]
        V[t + 1 :, idx] = 0.0

    return float((V[1] * np.exp(-r * dt)).mean())


def mc_call_convergence(grid: Array) -> Array:
    """Evaluate European call prices over a path-count grid."""

    estimates = [
        mc_european_call(100.0, 100.0, 0.02, 0.2, 1.0, n_paths=int(n))
        for n in grid
    ]
    return np.asarray(estimates, dtype=float)


def var_es(losses: Array, alpha: float = 0.99) -> tuple[float, float]:
    """Compute value-at-risk and expected shortfall from a loss sample."""

    var_level = float(np.quantile(losses, alpha))
    es_level = float(losses[losses >= var_level].mean())
    return var_level, es_level


def main() -> None:
    """Run a compact stochastic-simulation demo."""

    u, z = sample_uniform_normal()
    print(u.mean(), z.std())
    print(random_walk()[:3])
    print(gbm_paths()[:3, :3])
    print([arr.mean() for arr in gbm_terminal_levels()])
    print(cir_euler()[-1, :5])
    print(cir_exact()[:5])
    print(mc_european_call(100.0, 100.0, 0.02, 0.2, 1.0, n_paths=25_000))
    print(euro_put_bs(100.0, 100.0, 0.02, 0.2, 1.0))
    print(
        lsm_american_put(
            100.0, 100.0, 0.02, 0.2, 1.0, n_steps=50, n_paths=20_000
        )
    )
    grid = np.logspace(3, 6, num=6, dtype=int)
    print(mc_call_convergence(grid))
    pnl = (
        gbm_terminal_levels(
            s0=100.0,
            r=0.0,
            sigma=0.25,
            T=30.0 / 365.0,
            n_paths=50_000,
        )[0]
        - 100.0
    )
    print(var_es(-pnl))


if __name__ == '__main__':
    main()
