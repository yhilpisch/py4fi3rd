"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 29 - Derivatives Valuation.

Deterministic discounting building blocks for Monte Carlo valuation.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

__all__ = ["DiscountingModel", "FlatDiscounting", "discount_factors"]

FloatArray = NDArray[np.float64]


class DiscountingModel(Protocol):
    """
    Discounting model interface used in Monte Carlo valuation.

    The time argument is a year fraction (time-to-maturity) measured from 0.
    """

    def discount_factor(self, ttm: float) -> float: ...


@dataclass(frozen=True, slots=True)
class FlatDiscounting:
    """
    Constant continuously compounded short-rate discounting.
    """

    rate: float

    def discount_factor(self, ttm: float) -> float:
        if ttm < 0:
            raise ValueError("ttm must be non-negative")
        return float(math.exp(-self.rate * float(ttm)))


def discount_factors(model: DiscountingModel, times: FloatArray) -> FloatArray:
    """
    Vectorized discount factors for a NumPy array of year fractions.
    """

    if times.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if np.any(times < 0):
        raise ValueError("times must be non-negative")

    out = np.empty_like(times, dtype=float)
    for idx, ttm in enumerate(times):
        out[idx] = model.discount_factor(float(ttm))
    return out

