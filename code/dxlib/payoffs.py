"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 29 - Derivatives Valuation.

Payoff functions used in Part VI.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

__all__ = ["TerminalPayoff", "EuropeanCall", "EuropeanPut", "AmericanPut"]

FloatArray = NDArray[np.float64]


class TerminalPayoff(Protocol):
    """
    Payoff depending only on the terminal risk factor value.
    """

    def __call__(self, spot: FloatArray) -> FloatArray: ...


@dataclass(frozen=True, slots=True)
class EuropeanCall:
    strike: float

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError("strike must be positive")

    def __call__(self, spot: FloatArray) -> FloatArray:
        return np.maximum(spot - self.strike, 0.0)


@dataclass(frozen=True, slots=True)
class EuropeanPut:
    strike: float

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError("strike must be positive")

    def __call__(self, spot: FloatArray) -> FloatArray:
        return np.maximum(self.strike - spot, 0.0)


@dataclass(frozen=True, slots=True)
class AmericanPut:
    strike: float

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError("strike must be positive")

    def intrinsic_value(self, spot: FloatArray) -> FloatArray:
        return np.maximum(self.strike - spot, 0.0)

    def __call__(self, spot: FloatArray) -> FloatArray:
        return self.intrinsic_value(spot)

