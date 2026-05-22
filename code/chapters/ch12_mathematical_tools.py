"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 12 - Mathematical Tools.

This companion module collects typed helpers for the chapter's recurring
numerical patterns: regression, interpolation, optimisation,
integration, and symbolic work.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import integrate, optimize
from scipy.interpolate import CubicSpline

try:
    import sympy as sp
except ImportError:  # pragma: no cover - optional dependency
    sp = None


Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class RegressionResult:
    beta: Array
    fitted: Array


def term_structure(x: Array) -> Array:
    """Return the synthetic term-structure function used in the chapter."""

    return 0.5 * np.sin(x) + 0.1 * x


def noisy_observations(
    seed: int = 42,
    n: int = 50,
) -> tuple[Array, Array, Array]:
    """Generate a clean curve and a noisy sample on a regular grid."""

    rng = np.random.default_rng(seed=seed)
    x = np.linspace(0.0, 10.0, n)
    y_clean = term_structure(x)
    y_obs = y_clean + 0.05 * rng.standard_normal(n)
    return x, y_clean, y_obs


def fit_linear_regression(x: Array, y: Array) -> RegressionResult:
    """Fit an intercept-plus-slope least-squares model."""

    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return RegressionResult(beta=beta, fitted=X @ beta)


def fit_basis_regression(x: Array, y: Array) -> RegressionResult:
    """Fit a basis-function model with polynomial and sinusoidal terms."""

    X = np.column_stack([np.ones_like(x), x, x**2, np.sin(x)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return RegressionResult(beta=beta, fitted=X @ beta)


def fit_cubic_spline(x: Array, y: Array, x_fine: Array) -> Array:
    """Evaluate a cubic spline interpolant on a finer grid."""

    spline = CubicSpline(x, y)
    return spline(x_fine)


def solve_zero_coupon_yield(
    face: float,
    maturity: float,
    price: float,
) -> float:
    """Recover a continuously compounded yield from a zero-coupon price."""

    def pricing_error(yield_cc: float) -> float:
        return face * np.exp(-yield_cc * maturity) - price

    result = optimize.root_scalar(
        pricing_error,
        bracket=(0.0, 0.2),
        method='brentq',
    )
    return float(result.root)


def mean_variance_weights(
    mu: Array,
    cov: Array,
    target_return: float,
) -> Array:
    """Solve a two-asset long-only mean-variance problem."""

    def port_var(w: Array) -> float:
        return float(w @ cov @ w)

    def ret_constraint(w: Array) -> float:
        return float(w @ mu - target_return)

    result = optimize.minimize(
        port_var,
        np.repeat(1.0 / mu.shape[0], mu.shape[0]),
        method='SLSQP',
        bounds=tuple((0.0, 1.0) for _ in range(mu.shape[0])),
        constraints=(
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': ret_constraint},
        ),
    )
    if not result.success:
        raise RuntimeError('mean-variance optimization failed')
    return np.asarray(result.x, dtype=float)


def objective(p: npt.ArrayLike) -> npt.ArrayLike:
    """Evaluate the smooth two-parameter objective from the chapter."""

    x, y = p
    return np.sin(x) + 0.05 * x**2 + np.sin(y) + 0.05 * y**2


def optimize_objective() -> tuple[Array, float, Array, float]:
    """Combine a brute-force search with local refinement."""

    grid_ranges = ((-10.0, 10.0, 0.5), (-10.0, 10.0, 0.5))
    brute_min, f_min, _, _ = optimize.brute(
        objective,
        ranges=grid_ranges,
        full_output=True,
        finish=None,
    )
    local = optimize.minimize(objective, brute_min, method='Nelder-Mead')
    return (
        np.asarray(brute_min, dtype=float),
        float(f_min),
        np.asarray(local.x, dtype=float),
        float(local.fun),
    )


def normal_pdf(x: float) -> float:
    """Return the standard normal density."""

    return float((1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x * x))


def integrate_normal_density() -> tuple[float, float]:
    """Integrate the standard normal density over the real line."""

    value, error = integrate.quad(normal_pdf, -np.inf, np.inf)
    return float(value), float(error)


def black_scholes_integral_call(
    s0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
) -> tuple[float, float]:
    """Price a European call by numerical integration."""

    m = np.log(s0) + (r - 0.5 * sigma**2) * T
    v = sigma * np.sqrt(T)

    def lognormal_pdf(s: float) -> float:
        if s <= 0.0:
            return 0.0
        z = (np.log(s) - m) / v
        return float(
            (1.0 / (s * v * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * z * z)
        )

    def integrand(s: float) -> float:
        return float(np.exp(-r * T) * max(s - K, 0.0) * lognormal_pdf(s))

    value, error = integrate.quad(integrand, 0.0, 5.0 * s0)
    return float(value), float(error)


def symbolic_zero_coupon_yield() -> str | None:
    """Return the symbolic yield solution when SymPy is available."""

    if sp is None:
        return None
    y, T_sym, P_sym, F_sym = sp.symbols('y T P F', positive=True)
    bond_eq = sp.Eq(F_sym * sp.exp(-y * T_sym), P_sym)
    return str(sp.solve(bond_eq, y)[0])


def main() -> None:
    """Run a compact chapter-style demo."""

    x, y_clean, y_obs = noisy_observations()
    print(fit_linear_regression(x, y_obs).beta)
    print(fit_basis_regression(x, y_obs).beta)
    print(fit_cubic_spline(x, y_clean, np.linspace(0.0, 10.0, 8))[:5])
    print(solve_zero_coupon_yield(face=100.0, maturity=5.0, price=88.0))

    mu = np.array([0.06, 0.08], dtype=float)
    sigma = np.array([0.15, 0.25], dtype=float)
    rho = 0.4
    cov = np.array(
        [
            [sigma[0] ** 2, rho * sigma[0] * sigma[1]],
            [rho * sigma[0] * sigma[1], sigma[1] ** 2],
        ],
        dtype=float,
    )
    print(mean_variance_weights(mu, cov, target_return=0.07))
    print(optimize_objective())
    print(integrate_normal_density())
    print(black_scholes_integral_call(100.0, 100.0, 0.02, 0.2, 1.0))
    print(symbolic_zero_coupon_yield())


if __name__ == '__main__':
    main()
