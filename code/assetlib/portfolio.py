"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 21 - A Small Asset Management Library in Python.

Portfolio-construction utilities on top of MarketData and SignalEngine.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from assetlib.core import _ANNUALIZATION_FACTOR
    from assetlib.data import MarketData
else:
    from .core import _ANNUALIZATION_FACTOR
    from .data import MarketData


@dataclass
class RiskModel:
    """Annualized expected returns and covariance for a universe."""

    mu_annual: pd.Series
    cov_annual: pd.DataFrame


def estimate_mu_sigma(
    market_data: MarketData,
    universe: Sequence[str],
    freq: str = "D",
) -> RiskModel:
    """Estimate annualized expected returns and covariance."""

    if freq != "D":
        raise ValueError(
            f"unsupported frequency: {freq!r}",
        )  # only daily data is supported

    rets = market_data.returns(universe)  # daily returns
    mu = rets.mean()  # daily mean
    cov = rets.cov()  # daily covariance
    ann = _ANNUALIZATION_FACTOR  # trading days per year
    mu_annual = (1.0 + mu) ** ann - 1.0  # compounded
    cov_annual = cov * ann  # scaled covariance
    return RiskModel(mu_annual=mu_annual, cov_annual=cov_annual)  # bundle


def equal_weight(universe: Sequence[str]) -> pd.Series:
    n = len(universe)
    if n == 0:
        raise ValueError("universe must not be empty")
    w = np.repeat(1.0 / n, n)  # equal share
    return pd.Series(w, index=list(universe), name="equal_weight")


def _project_capped_simplex(
    weights: np.ndarray,
    cap: float,
) -> np.ndarray:
    """Project onto {w >= 0, sum(w) = 1, w_i <= cap}."""

    w = np.asarray(weights, dtype=float).copy()
    w = np.clip(w, 0.0, None)  # enforce long-only
    if cap * len(w) < 1.0:
        raise ValueError("cap too low to allow full investment")
    remaining = np.ones(len(w), dtype=bool)
    out = np.zeros(len(w), dtype=float)
    mass = 1.0
    while True:  # iterative cap redistribution
        if remaining.sum() == 0:
            return out
        s = w[remaining].sum()
        if s == 0.0:
            out[remaining] = mass / remaining.sum()
            return out
        scaled = w[remaining] * (mass / s)
        over = scaled > cap
        if not np.any(over):
            out[remaining] = scaled
            return out
        idx_over = np.where(remaining)[0][over]
        out[idx_over] = cap  # pin assets at the cap
        remaining[idx_over] = False
        mass = 1.0 - out[~remaining].sum()


def signal_tilt(
    signal: pd.Series,
    long_only: bool = True,
    cap_per_asset: float | None = None,
) -> pd.Series:
    """Map a cross-sectional signal into portfolio weights."""

    if signal.isna().all():
        raise ValueError("signal contains only NaNs")

    std = signal.std(ddof=0)  # cross-sectional dispersion
    if std:
        z = (signal - signal.mean()) / std  # z-scores
    else:
        z = signal * 0.0  # flat when no variation
    if long_only:
        z = z.clip(lower=0.0)
    if z.abs().sum() == 0.0:
        w = np.repeat(1.0 / len(z), len(z))  # equal-weight fallback
        weights = pd.Series(w, index=signal.index)
    else:
        weights = z / z.abs().sum()

    if cap_per_asset is not None:
        if cap_per_asset <= 0.0:
            raise ValueError("cap_per_asset must be positive")
        if long_only:
            capped = _project_capped_simplex(
                weights.to_numpy(),
                cap_per_asset,
            )  # respects the single-name cap
            weights = pd.Series(capped, index=weights.index)
        else:
            weights = weights.clip(upper=cap_per_asset)

    return weights.rename("weights")


def gmv_weights(cov_annual: pd.DataFrame) -> pd.Series:
    """Global minimum-variance weights given an annualized covariance matrix."""

    sigma = cov_annual.values  # covariance matrix
    n = sigma.shape[0]
    ones = np.ones(n)  # unit vector
    inv_sigma_ones = np.linalg.solve(sigma, ones)  # Σ^{-1}1
    w = inv_sigma_ones / (ones @ inv_sigma_ones)  # normalized GMV weights
    return pd.Series(w, index=cov_annual.index, name="gmv")
