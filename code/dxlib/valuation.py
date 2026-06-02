"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 29 - Derivatives Valuation.

Monte Carlo valuation helpers.

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

from .discounting import DiscountingModel
from .payoffs import TerminalPayoff
from .processes import PathSimulator, build_time_grid

__all__ = ["EuropeanMCPricer"]


@dataclass(frozen=True, slots=True)
class EuropeanMCPricer:
    """
    Terminal-payoff Monte Carlo valuation.
    """

    process: PathSimulator
    payoff: TerminalPayoff
    discounting: DiscountingModel
    maturity: float
    steps: int
    paths: int

    def __post_init__(self) -> None:
        if self.maturity <= 0:
            raise ValueError("maturity must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.paths <= 0:
            raise ValueError("paths must be positive")

    def value(self, spot: float) -> tuple[float, float]:
        time_grid = build_time_grid(self.maturity, self.steps)
        paths = self.process.simulate_paths(
            spot=spot,
            time_grid=time_grid,
            paths=self.paths,
        )
        payoff = self.payoff(paths[:, -1])
        df = self.discounting.discount_factor(self.maturity)
        pv = df * payoff
        price = float(np.mean(pv))
        stderr = float(np.std(pv, ddof=1) / math.sqrt(self.paths))
        return price, stderr
