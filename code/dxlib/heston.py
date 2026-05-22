"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 31 - Market-Based Valuation.

Heston model helpers for Monte Carlo pricing and calibration.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

if __name__ == "__main__" and __package__ is None:
    package_dir = Path(__file__).resolve().parent
    sys.path = [
        path
        for path in sys.path
        if Path(path or ".").resolve() != package_dir
    ]
    sys.path.insert(0, str(package_dir.parent))
    __package__ = "dxlib"

from .processes import HestonModel, build_time_grid

__all__ = ["HestonParams", "simulate_heston_spot_paths", "mc_call_prices"]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class HestonParams:
    kappa: float
    theta: float
    vol_of_vol: float
    rho: float
    v0: float

    def __post_init__(self) -> None:
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        if self.theta <= 0:
            raise ValueError("theta must be positive")
        if self.vol_of_vol < 0:
            raise ValueError("vol_of_vol must be non-negative")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must be in [-1, 1]")
        if self.v0 <= 0:
            raise ValueError("v0 must be positive")


def simulate_heston_spot_paths(
    *,
    spot: float,
    rate: float,
    maturity: float,
    steps: int,
    paths: int,
    params: HestonParams,
    antithetic: bool = True,
    moment_matching: bool = False,
    seed: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    """
    Simulate spot and variance paths under the Heston model.
    """

    grid = build_time_grid(maturity=float(maturity), steps=int(steps))
    model = HestonModel(
        kappa=float(params.kappa),
        theta=float(params.theta),
        vol_of_vol=float(params.vol_of_vol),
        rho=float(params.rho),
        drift=float(rate),
        seed=seed,
        antithetic=bool(antithetic),
        moment_matching=bool(moment_matching),
    )
    return model.simulate_paths(
        spot=float(spot),
        variance=float(params.v0),
        time_grid=grid,
        paths=int(paths),
    )


def mc_call_prices(
    spot_paths: FloatArray,
    strikes: FloatArray,
    *,
    discount_factor: float,
    forward: float | None = None,
) -> FloatArray:
    """
    Monte Carlo prices for European calls from spot paths.
    """

    if spot_paths.ndim != 2:
        raise ValueError("spot_paths must be two-dimensional")
    if strikes.ndim != 1:
        raise ValueError("strikes must be one-dimensional")

    df = float(discount_factor)
    if df <= 0:
        raise ValueError("discount_factor must be positive")

    terminal = spot_paths[:, -1].astype(float, copy=False)
    payoffs = np.maximum(terminal[:, None] - strikes[None, :], 0.0)

    mean_payoffs = np.mean(payoffs, axis=0)
    if forward is None:
        prices = df * mean_payoffs
        return prices.astype(float, copy=False)

    fwd = float(forward)
    y = terminal
    y_mean = float(np.mean(y))
    y_center = y - y_mean
    var_y = float(np.mean(y_center**2))
    if var_y <= 0.0:
        prices = df * mean_payoffs
        return prices.astype(float, copy=False)

    x_center = payoffs - mean_payoffs[None, :]
    cov_xy = np.mean(x_center * y_center[:, None], axis=0)
    beta = cov_xy / var_y
    adj_means = mean_payoffs + beta * (fwd - y_mean)
    prices = df * adj_means
    return prices.astype(float, copy=False)
