"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 28 - Simulation of Financial Models.

Risk-factor simulation models used throughout Part VI.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

# Run-from-source fallback: allows executing this module directly
# (python code/dxlib/curves.py). It places the parent directory (code/)
# on sys.path and sets __package__ so that relative imports resolve.
# When dxlib is imported as a package from the project root, the guard
# is False and the block is skipped entirely.
if __name__ == "__main__" and __package__ is None:
    package_dir = Path(__file__).resolve().parent
    sys.path = [
        path
        for path in sys.path
        if Path(path or ".").resolve() != package_dir
    ]
    sys.path.insert(0, str(package_dir.parent))
    __package__ = "dxlib"

from .random import standard_normals

__all__ = [
    "PathSimulator",
    "build_time_grid",
    "GeometricBrownianMotion",
    "JumpDiffusion",
    "HestonModel",
    "CIRShortRate",
]

from .types import FloatArray


class PathSimulator(Protocol):
    """
    Protocol for single-factor simulators used by valuation routines.
    """

    def simulate_paths(
        self,
        spot: float,
        time_grid: FloatArray,
        paths: int,
    ) -> FloatArray: ...


def build_time_grid(maturity: float, steps: int) -> FloatArray:
    """
    Build a uniform time grid from 0 to maturity (inclusive).
    """

    if maturity <= 0:
        raise ValueError("maturity must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
    grid = np.linspace(0.0, float(maturity), int(steps) + 1)
    return grid.astype(float, copy=False)


def _validate_grid(grid: FloatArray) -> None:
    if grid.ndim != 1:
        raise ValueError("time_grid must be one-dimensional")
    if grid.size < 2:
        raise ValueError("time_grid must contain at least two points")
    if float(grid[0]) != 0.0:
        raise ValueError("time_grid must start at 0")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("time_grid must be strictly increasing")


@dataclass(frozen=True, slots=True)
class GeometricBrownianMotion:
    """
    Geometric Brownian motion (GBM) process for equity prices.

    The process is dS_t = mu * S_t * dt + sigma * S_t * dW_t, and the
    simulation uses the exact log-normal step.
    """

    drift: float
    volatility: float
    seed: int | None = None
    antithetic: bool = True
    moment_matching: bool = True

    def __post_init__(self) -> None:
        if self.volatility < 0:
            raise ValueError("volatility must be non-negative")

    def simulate_paths(
        self,
        spot: float,
        time_grid: FloatArray,
        paths: int,
    ) -> FloatArray:
        _validate_grid(time_grid)
        if spot <= 0:
            raise ValueError("spot must be positive")
        if paths <= 0:
            raise ValueError("paths must be positive")

        step_sizes = np.diff(time_grid)
        n_steps = step_sizes.size
        shocks = standard_normals(
            (n_steps, paths),
            seed=self.seed,
            antithetic=self.antithetic,
            moment_matching=self.moment_matching,
        )

        out = np.empty((paths, time_grid.size), dtype=float)
        out[:, 0] = float(spot)

        for step_index, dt in enumerate(step_sizes):
            drift_term = (self.drift - 0.5 * self.volatility**2) * dt
            diffusion_term = (
                self.volatility * math.sqrt(float(dt)) * shocks[step_index]
            )
            out[:, step_index + 1] = out[:, step_index] * np.exp(
                drift_term + diffusion_term
            )

        return out


@dataclass(frozen=True, slots=True)
class JumpDiffusion:
    """
    Merton (1976) jump diffusion with log-normal jumps.

    Jump arrivals follow a Poisson process with intensity lambda,
    and log jump sizes are normal.
    """

    drift: float
    volatility: float
    jump_intensity: float
    jump_mean: float
    jump_std: float
    seed: int | None = None
    antithetic: bool = True
    moment_matching: bool = True

    def __post_init__(self) -> None:
        if self.volatility < 0:
            raise ValueError("volatility must be non-negative")
        if self.jump_intensity < 0:
            raise ValueError("jump_intensity must be non-negative")
        if self.jump_std < 0:
            raise ValueError("jump_std must be non-negative")

    def simulate_paths(
        self,
        spot: float,
        time_grid: FloatArray,
        paths: int,
    ) -> FloatArray:
        _validate_grid(time_grid)
        if spot <= 0:
            raise ValueError("spot must be positive")
        if paths <= 0:
            raise ValueError("paths must be positive")

        diffusion_seed = self.seed
        jump_seed = None if self.seed is None else self.seed + 1
        rng = np.random.default_rng(jump_seed)
        step_sizes = np.diff(time_grid)
        n_steps = step_sizes.size
        shocks = standard_normals(
            (n_steps, paths),
            seed=diffusion_seed,
            antithetic=self.antithetic,
            moment_matching=self.moment_matching,
        )

        jump_compensation = self.jump_intensity * (
            math.exp(self.jump_mean + 0.5 * self.jump_std**2) - 1.0
        )

        out = np.empty((paths, time_grid.size), dtype=float)
        out[:, 0] = float(spot)

        for step_index, dt in enumerate(step_sizes):
            poisson = rng.poisson(self.jump_intensity * float(dt), size=paths)
            jump_normals = rng.standard_normal(size=paths)
            if self.antithetic:
                half = paths // 2
                jump_normals[half:] = -jump_normals[:half]

            sqrt_poisson = np.sqrt(poisson.astype(float, copy=False))
            jump_exponent = (
                poisson * self.jump_mean
                + self.jump_std * sqrt_poisson * jump_normals
            )
            jump_factor = np.exp(
                jump_exponent
            )  # product of N iid log-normal jumps

            drift_term = (
                self.drift - jump_compensation - 0.5 * self.volatility**2
            ) * float(dt)
            diffusion_term = (
                self.volatility * math.sqrt(float(dt)) * shocks[step_index]
            )
            out[:, step_index + 1] = (
                out[:, step_index]
                * np.exp(drift_term + diffusion_term)
                * jump_factor
            )

        return out


@dataclass(frozen=True, slots=True)
class HestonModel:
    """
    Heston (1993) stochastic volatility model (Euler-style discretization).

    The model evolves spot S_t and variance v_t
    with correlated Brownian motions.
    """

    kappa: float
    theta: float
    vol_of_vol: float
    rho: float
    drift: float
    seed: int | None = None
    antithetic: bool = True
    moment_matching: bool = True

    def __post_init__(self) -> None:
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        if self.theta <= 0:
            raise ValueError("theta must be positive")
        if self.vol_of_vol < 0:
            raise ValueError("vol_of_vol must be non-negative")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must be in [-1, 1]")

    def simulate_paths(
        self,
        spot: float,
        variance: float,
        time_grid: FloatArray,
        paths: int,
    ) -> tuple[FloatArray, FloatArray]:
        _validate_grid(time_grid)
        if spot <= 0:
            raise ValueError("spot must be positive")
        if variance <= 0:
            raise ValueError("variance must be positive")
        if paths <= 0:
            raise ValueError("paths must be positive")

        step_sizes = np.diff(time_grid)
        n_steps = step_sizes.size
        normals = standard_normals(
            (n_steps, 2, paths),
            seed=self.seed,
            antithetic=self.antithetic,
            moment_matching=self.moment_matching,
        )

        out_spot = np.empty((paths, time_grid.size), dtype=float)
        out_var = np.empty_like(out_spot)
        out_spot[:, 0] = float(spot)
        out_var[:, 0] = float(variance)

        sqrt_one_minus_rho2 = math.sqrt(1.0 - self.rho**2)

        for step_index, dt in enumerate(step_sizes):
            z1 = normals[step_index, 0, :]
            z2 = normals[step_index, 1, :]
            w1 = z1
            w2 = self.rho * z1 + sqrt_one_minus_rho2 * z2

            prev_var = out_var[:, step_index]
            prev_spot = out_spot[:, step_index]

            prev_var_pos = np.maximum(prev_var, 0.0)
            sqrt_var = np.sqrt(prev_var_pos)

            drift_var = self.kappa * (self.theta - prev_var_pos) * float(dt)
            diffusion_var = (
                self.vol_of_vol * sqrt_var * math.sqrt(float(dt)) * w2
            )
            new_var = np.maximum(prev_var_pos + drift_var + diffusion_var, 0.0)
            out_var[:, step_index + 1] = new_var

            drift_spot = (self.drift - 0.5 * prev_var_pos) * float(dt)
            diffusion_spot = sqrt_var * math.sqrt(float(dt)) * w1
            out_spot[:, step_index + 1] = prev_spot * np.exp(
                drift_spot + diffusion_spot
            )

        return out_spot, out_var


@dataclass(frozen=True, slots=True)
class CIRShortRate:
    """
    Cox-Ingersoll-Ross (1985) short-rate model (Euler discretization with
    truncation).
    """

    kappa: float
    theta: float
    sigma: float
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        if self.theta <= 0:
            raise ValueError("theta must be positive")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")

    def simulate_paths(
        self,
        rate0: float,
        time_grid: FloatArray,
        paths: int,
    ) -> FloatArray:
        _validate_grid(time_grid)
        if rate0 < 0:
            raise ValueError("rate0 must be non-negative")
        if paths <= 0:
            raise ValueError("paths must be positive")

        rng = np.random.default_rng(self.seed)
        step_sizes = np.diff(time_grid)
        out = np.empty((paths, time_grid.size), dtype=float)
        out[:, 0] = float(rate0)

        for step_index, dt in enumerate(step_sizes):
            z = rng.standard_normal(paths)
            prev = out[:, step_index]
            prev_pos = np.maximum(prev, 0.0)
            drift = self.kappa * (self.theta - prev_pos) * float(dt)
            diffusion = (
                self.sigma * np.sqrt(prev_pos) * math.sqrt(float(dt)) * z
            )
            out[:, step_index + 1] = np.maximum(
                prev_pos + drift + diffusion,
                0.0,
            )

        return out
