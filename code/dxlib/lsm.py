"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 29 - Derivatives Valuation.

Least-squares Monte Carlo (Longstaff-Schwartz) valuation for American options.

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

from .discounting import DiscountingModel, discount_factors
from .payoffs import AmericanPut
from .processes import PathSimulator, build_time_grid

__all__ = ["AmericanPutLSM", "lsm_american_put_from_paths"]

FloatArray = NDArray[np.float64]


def _poly_basis(x: FloatArray, degree: int) -> FloatArray:
    if degree < 0:
        raise ValueError("degree must be non-negative")
    columns = [np.ones_like(x)]
    for pow_ in range(1, degree + 1):
        columns.append(x**pow_)
    return np.column_stack(columns).astype(float, copy=False)


def lsm_american_put_from_paths(
    spot_paths: FloatArray,
    *,
    strike: float,
    discounting: DiscountingModel,
    time_grid: FloatArray,
    basis_degree: int = 2,
    min_itm: int = 200,
) -> dict[str, object]:
    """
    LSM valuation for an American put using pre-simulated spot paths.
    """

    if spot_paths.ndim != 2:
        raise ValueError("spot_paths must be two-dimensional")
    if time_grid.ndim != 1:
        raise ValueError("time_grid must be one-dimensional")
    if spot_paths.shape[1] != time_grid.size:
        raise ValueError("spot_paths and time_grid must align")
    if strike <= 0:
        raise ValueError("strike must be positive")
    if basis_degree < 1:
        raise ValueError("basis_degree must be at least 1")
    if min_itm < 10:
        raise ValueError("min_itm must be at least 10")

    payoff = AmericanPut(strike=float(strike))
    intrinsic = payoff.intrinsic_value(spot_paths)
    cashflow = intrinsic[:, -1].astype(float, copy=True)

    df_grid = discount_factors(discounting, time_grid)
    df_step = df_grid[1:] / df_grid[:-1]

    steps = int(time_grid.size - 1)
    boundary = np.full(time_grid.size, np.nan, dtype=float)
    exercise_index = np.full(spot_paths.shape[0], steps, dtype=int)

    for step in range(steps - 1, 0, -1):
        cashflow *= float(df_step[step])

        spot_t = spot_paths[:, step]
        intrinsic_t = intrinsic[:, step]
        itm = intrinsic_t > 0.0
        if np.count_nonzero(itm) < min_itm:
            continue

        x = (spot_t[itm] / float(strike)).astype(float, copy=False)
        y = cashflow[itm]
        x_mat = _poly_basis(x, degree=basis_degree)
        beta, *_ = np.linalg.lstsq(x_mat, y, rcond=None)
        continuation = x_mat @ beta

        exercise = intrinsic_t[itm] > continuation
        if np.any(exercise):
            exercised_spots = spot_t[itm][exercise]
            boundary[step] = float(np.max(exercised_spots))

        new_cashflow = cashflow[itm].copy()
        new_cashflow[exercise] = intrinsic_t[itm][exercise]
        cashflow[itm] = new_cashflow

        exercise_paths = np.flatnonzero(itm)[exercise]
        exercise_index[exercise_paths] = step

    pv0 = cashflow * float(df_step[0])
    price = float(np.mean(pv0))
    stderr = float(np.std(pv0, ddof=1) / math.sqrt(spot_paths.shape[0]))

    return {
        "price": price,
        "stderr": stderr,
        "time_grid": time_grid,
        "exercise_boundary": boundary,
        "exercise_index": exercise_index,
    }


@dataclass(frozen=True, slots=True)
class AmericanPutLSM:
    """
    Least-squares Monte Carlo (LSM) valuation for an American put option.

    The implementation follows Longstaff & Schwartz (2001) and uses polynomial
    basis functions in the normalized state variable latexmath:[S_t / K].
    """

    process: PathSimulator
    payoff: AmericanPut
    discounting: DiscountingModel
    maturity: float
    steps: int
    paths: int
    basis_degree: int = 2
    min_itm: int = 200

    def __post_init__(self) -> None:
        if self.maturity <= 0:
            raise ValueError("maturity must be positive")
        if self.steps <= 1:
            raise ValueError("steps must be at least 2")
        if self.paths <= 0:
            raise ValueError("paths must be positive")
        if self.basis_degree < 1:
            raise ValueError("basis_degree must be at least 1")
        if self.min_itm < 10:
            raise ValueError("min_itm must be at least 10")

    def value(self, spot: float) -> dict[str, object]:
        strike = self.payoff.strike
        time_grid = build_time_grid(self.maturity, self.steps)
        spot_paths = self.process.simulate_paths(
            spot=spot,
            time_grid=time_grid,
            paths=self.paths,
        )

        intrinsic = self.payoff.intrinsic_value(spot_paths)
        cashflow = intrinsic[:, -1].astype(float, copy=True)

        df_grid = discount_factors(self.discounting, time_grid)
        df_step = df_grid[1:] / df_grid[:-1]

        boundary = np.full(time_grid.size, np.nan, dtype=float)
        exercise_index = np.full(self.paths, self.steps, dtype=int)

        for step in range(self.steps - 1, 0, -1):
            cashflow *= float(df_step[step])

            spot_t = spot_paths[:, step]
            intrinsic_t = intrinsic[:, step]
            itm = intrinsic_t > 0.0
            if np.count_nonzero(itm) < self.min_itm:
                continue

            x = (spot_t[itm] / strike).astype(float, copy=False)
            y = cashflow[itm]
            x_mat = _poly_basis(x, degree=self.basis_degree)
            beta, *_ = np.linalg.lstsq(x_mat, y, rcond=None)
            continuation = x_mat @ beta

            exercise = intrinsic_t[itm] > continuation
            if np.any(exercise):
                exercised_spots = spot_t[itm][exercise]
                boundary[step] = float(np.max(exercised_spots))

            new_cashflow = cashflow[itm].copy()
            new_cashflow[exercise] = intrinsic_t[itm][exercise]
            cashflow[itm] = new_cashflow

            exercise_paths = np.flatnonzero(itm)[exercise]
            exercise_index[exercise_paths] = step

        pv0 = cashflow * float(df_step[0])
        price = float(np.mean(pv0))
        stderr = float(np.std(pv0, ddof=1) / math.sqrt(self.paths))

        out: dict[str, object] = {
            "price": price,
            "stderr": stderr,
            "time_grid": time_grid,
            "exercise_boundary": boundary,
            "exercise_index": exercise_index,
        }
        return out
