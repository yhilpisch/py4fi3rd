"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 27 - Valuation Framework.

Discount curve building blocks.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import datetime as dt
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

if __name__ == "__main__" and __package__ is None:
    package_dir = Path(__file__).resolve().parent
    sys.path = [
        path
        for path in sys.path
        if Path(path or ".").resolve() != package_dir
    ]
    sys.path.insert(0, str(package_dir.parent))
    __package__ = "dxlib"

from .time import time_to_maturity

__all__ = ["DiscountCurve", "ConstantShortRateCurve", "InterpolatedZeroCurve"]


class DiscountCurve(Protocol):
    """
    Minimal discounting interface used throughout Part VI.

    Deterministic curves and stochastic discounting models can both implement
    this protocol.
    """

    reference_date: dt.date | dt.datetime

    def discount_factor(self, target: dt.date | dt.datetime) -> float: ...


@dataclass(frozen=True, slots=True)
class ConstantShortRateCurve:
    """
    Constant continuously compounded short-rate curve.
    """

    name: str
    reference_date: dt.date | dt.datetime
    rate: float
    day_count: float = 365.0

    def __post_init__(self) -> None:
        if self.day_count <= 0:
            raise ValueError("day_count must be positive")

    def discount_factor(self, target: dt.date | dt.datetime) -> float:
        ttm = time_to_maturity(
            self.reference_date,
            target,
            day_count=self.day_count,
        )
        return float(math.exp(-self.rate * ttm))


@dataclass(frozen=True)
class InterpolatedZeroCurve:
    """
    Deterministic zero curve represented by (date, zero_rate) nodes.

    Linear interpolation is applied to continuously compounded zero
    rates.
    """

    name: str
    reference_date: dt.date | dt.datetime
    nodes: Sequence[tuple[dt.date | dt.datetime, float]]
    day_count: float = 365.0
    extrapolate: str = "flat"

    def __post_init__(self) -> None:
        if self.day_count <= 0:
            raise ValueError("day_count must be positive")
        if len(self.nodes) < 2:
            raise ValueError(
                "InterpolatedZeroCurve requires at least two nodes"
            )
        mode = str(self.extrapolate).strip().lower()
        if mode not in {"flat", "error"}:
            raise ValueError("extrapolate must be 'flat' or 'error'")
        times, _ = self._times_and_rates()
        if len(np.unique(times)) != times.size:
            raise ValueError("curve node maturities must be unique")

    def _times_and_rates(self) -> tuple[np.ndarray, np.ndarray]:
        ordered = sorted(self.nodes, key=lambda pair: pair[0])
        dates, rates = zip(*ordered)
        times = np.array(
            [
                time_to_maturity(
                    self.reference_date,
                    d,
                    day_count=self.day_count,
                )
                for d in dates
            ],
            dtype=float,
        )
        return times, np.array(rates, dtype=float)

    def zero_rate(self, target: dt.date | dt.datetime) -> float:
        if target == self.reference_date:
            return 0.0
        times, rates = self._times_and_rates()
        t = time_to_maturity(
            self.reference_date,
            target,
            day_count=self.day_count,
        )
        if t <= 0:
            raise ValueError("target must be after reference_date")
        mode = str(self.extrapolate).strip().lower()
        if mode == "error" and (t < float(times[0]) or t > float(times[-1])):
            raise ValueError("target is outside curve node range")
        r = float(np.interp(t, times, rates))
        return r

    def discount_factor(self, target: dt.date | dt.datetime) -> float:
        if target == self.reference_date:
            return 1.0
        t = time_to_maturity(
            self.reference_date,
            target,
            day_count=self.day_count,
        )
        r = self.zero_rate(target)
        return float(math.exp(-r * t))
