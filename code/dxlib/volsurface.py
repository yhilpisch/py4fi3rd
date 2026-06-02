"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 31 - Market-Based Valuation.

Simple implied-volatility surface built from a small set of smiles.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["ImpliedVolSurface"]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ImpliedVolSurface:
    """
    Bilinear interpolation on (time-to-maturity, moneyness) nodes.
    """

    maturities: FloatArray
    moneyness_grid: dict[float, FloatArray]
    iv_grid: dict[float, FloatArray]
    extrapolate: str = "flat"

    def __post_init__(self) -> None:
        mode = str(self.extrapolate).strip().lower()
        if mode not in {"flat", "error"}:
            raise ValueError("extrapolate must be 'flat' or 'error'")
        if self.maturities.ndim != 1 or self.maturities.size == 0:
            raise ValueError(
                "maturities must be a non-empty one-dimensional array"
            )
        if np.any(np.diff(self.maturities) <= 0):
            raise ValueError("maturities must be strictly increasing")
        for maturity in self.maturities:
            t = float(maturity)
            mny = self.moneyness_grid[t]
            iv = self.iv_grid[t]
            if mny.ndim != 1 or iv.ndim != 1 or mny.size != iv.size:
                raise ValueError("moneyness and iv grids must be aligned")
            if mny.size == 0 or np.any(np.diff(mny) <= 0):
                raise ValueError("moneyness nodes must be strictly increasing")

    @classmethod
    def from_frame(
        cls,
        frame,
        *,
        maturity_col: str = "TTM",
        moneyness_col: str = "MNY",
        iv_col: str = "IMPL_VOL",
        extrapolate: str = "flat",
    ) -> "ImpliedVolSurface":
        maturities = np.sort(frame[maturity_col].unique()).astype(float)
        mny_grid: dict[float, FloatArray] = {}
        iv_grid: dict[float, FloatArray] = {}
        for t in maturities:
            sub = frame[frame[maturity_col] == t].sort_values(moneyness_col)
            mny = sub[moneyness_col].to_numpy(dtype=float)
            iv = sub[iv_col].to_numpy(dtype=float)
            mny_grid[float(t)] = mny
            iv_grid[float(t)] = iv
        return cls(
            maturities=maturities,
            moneyness_grid=mny_grid,
            iv_grid=iv_grid,
            extrapolate=extrapolate,
        )

    def _interp_smile(self, maturity: float, moneyness: float) -> float:
        mode = str(self.extrapolate).strip().lower()
        mny = self.moneyness_grid[maturity]
        iv = self.iv_grid[maturity]
        if mode == "error":
            if moneyness < float(mny[0]) or moneyness > float(mny[-1]):
                raise ValueError("moneyness is outside surface node range")
        return float(np.interp(moneyness, mny, iv))

    def vol(self, maturity: float, moneyness: float) -> float:
        t = float(maturity)
        x = float(moneyness)
        if t <= 0:
            raise ValueError("maturity must be positive")
        if x <= 0:
            raise ValueError("moneyness must be positive")

        mode = str(self.extrapolate).strip().lower()
        t_grid = self.maturities
        if t <= float(t_grid[0]):
            t0 = float(t_grid[0])
            if mode == "error" and t < t0:
                raise ValueError("maturity is outside surface node range")
            return self._interp_smile(t0, x)
        if t >= float(t_grid[-1]):
            t1 = float(t_grid[-1])
            if mode == "error" and t > t1:
                raise ValueError("maturity is outside surface node range")
            return self._interp_smile(t1, x)

        idx = int(np.searchsorted(t_grid, t))
        t_lo = float(t_grid[idx - 1])
        t_hi = float(t_grid[idx])
        iv_lo = self._interp_smile(t_lo, x)
        iv_hi = self._interp_smile(t_hi, x)
        weight = (t - t_lo) / (t_hi - t_lo)
        return float((1.0 - weight) * iv_lo + weight * iv_hi)
