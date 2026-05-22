"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 28 - Simulation of Financial Models.

Random number helpers for Monte Carlo simulation.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["standard_normals"]

FloatArray = NDArray[np.float64]


def standard_normals(
    shape: tuple[int, ...],
    *,
    seed: int | None = None,
    antithetic: bool = False,
    moment_matching: bool = False,
) -> FloatArray:
    """
    Generate standard normal random numbers with optional variance reduction.

    Parameters
    ----------
    shape:
        Output shape of the array.
    seed:
        Seed for reproducibility. ``None`` uses NumPy's default RNG seeding.
    antithetic:
        If ``True``, generate antithetic pairs along the last axis.
        This requires
        the last dimension to be even.
    moment_matching:
        If ``True``, shift and scale the sample so it has mean 0 and standard
        deviation 1.

    Returns
    -------
    ndarray
        Array of standard normal variates.
    """

    if any(dim <= 0 for dim in shape):
        raise ValueError("All shape dimensions must be positive")

    rng = np.random.default_rng(seed=seed)
    out = rng.standard_normal(size=shape).astype(float, copy=False)

    if antithetic:
        if len(shape) == 0:
            raise ValueError(
                "antithetic sampling requires an array, not a scalar"
            )
        last = shape[-1]
        if last % 2 != 0:
            raise ValueError(
                "antithetic sampling requires an even last dimension"
            )
        half = last // 2
        base = out[..., :half]
        out[..., half:] = -base

    if moment_matching:
        mean = float(out.mean())
        std = float(out.std(ddof=0))
        if std == 0.0:
            raise ValueError(
                "moment matching failed: sample standard deviation is zero"
            )
        out = (out - mean) / std

    return out
