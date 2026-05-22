"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 21 - A Small Asset Management Library in Python.

Portfolio-construction utilities on top of MarketData and SignalEngine.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from assetlib.data import MarketData
else:
    from .data import MarketData


def _annualization_factor(freq: str = "D") -> float:
    if freq == "D":
        return 252.0  # trading days per year
    msg = f"unsupported frequency: {freq!r}"
    raise ValueError(msg)


@dataclass
class RiskModel:
    mu_annual: pd.Series
    cov_annual: pd.DataFrame


def estimate_mu_sigma(
    market_data: MarketData,
    universe: Sequence[str],
    freq: str = "D",
) -> RiskModel:
    """Estimate annualized expected returns and covariance."""

    rets = market_data.returns(universe)  # daily returns
    mu = rets.mean()  # daily mean
    cov = rets.cov()  # daily covariance
    ann = _annualization_factor(freq)  # annualization factor
    mu_annual = (1.0 + mu) ** ann - 1.0  # compounded
    cov_annual = cov * ann  # scaled covariance
    return RiskModel(mu_annual=mu_annual, cov_annual=cov_annual)  # bundle


def equal_weight(universe: Sequence[str]) -> pd.Series:
    n = len(universe)
    if n == 0:
        raise ValueError("universe must not be empty")
    w = np.repeat(1.0 / n, n)  # equal share
    return pd.Series(w, index=list(universe), name="equal_weight")


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
        weights = weights.clip(upper=cap_per_asset)  # single-name cap
        total = weights.sum()
        if total == 0.0:
            raise ValueError("all weights clipped to zero by cap_per_asset")
        weights = weights / total  # renormalize

    return weights.rename("weights")


def gmv_weights(cov_annual: pd.DataFrame) -> pd.Series:
    """Global minimum-variance weights given an annualized covariance matrix."""

    sigma = cov_annual.values  # covariance matrix
    n = sigma.shape[0]
    ones = np.ones(n)  # unit vector
    inv_sigma_ones = np.linalg.solve(sigma, ones)  # Σ^{-1}1
    w = inv_sigma_ones / (ones @ inv_sigma_ones)  # normalized GMV weights
    return pd.Series(w, index=cov_annual.index, name="gmv")
