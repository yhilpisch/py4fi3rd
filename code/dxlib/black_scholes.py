"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 31 - Market-Based Valuation.

Black-Scholes (Black-76 forward form) pricing and implied volatility.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import math

__all__ = [
    "norm_cdf",
    "norm_pdf",
    "bs_price_forward",
    "implied_vol_forward",
]


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    x = float(x)
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_price_forward(
    forward: float,
    strike: float,
    maturity: float,
    volatility: float,
    *,
    option_type: str = "call",
    discount_factor: float = 1.0,
) -> float:
    """
    Black-76 price for an option on a forward with deterministic discounting.
    """

    fwd = float(forward)
    k = float(strike)
    t = float(maturity)
    vol = float(volatility)
    df = float(discount_factor)
    opt = str(option_type).strip().lower()

    if fwd <= 0 or k <= 0:
        raise ValueError("forward and strike must be positive")
    if t < 0:
        raise ValueError("maturity must be non-negative")
    if vol < 0:
        raise ValueError("volatility must be non-negative")
    if df <= 0:
        raise ValueError("discount_factor must be positive")
    if opt not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    if t == 0.0 or vol == 0.0:
        intrinsic = max(fwd - k, 0.0)
        if opt == "put":
            intrinsic = max(k - fwd, 0.0)
        return df * intrinsic

    sqrt_t = math.sqrt(t)
    sig_sqrt = vol * sqrt_t
    d1 = (math.log(fwd / k) + 0.5 * vol * vol * t) / sig_sqrt
    d2 = d1 - sig_sqrt

    if opt == "call":
        value = df * (fwd * norm_cdf(d1) - k * norm_cdf(d2))
    else:
        value = df * (k * norm_cdf(-d2) - fwd * norm_cdf(-d1))

    return float(value)


def implied_vol_forward(
    price: float,
    forward: float,
    strike: float,
    maturity: float,
    *,
    option_type: str = "call",
    discount_factor: float = 1.0,
    tol: float = 1.0e-8,
    max_iter: int = 100,
) -> float:
    """
    Implied volatility for Black-76 with deterministic discounting.

    Uses a robust bisection method with an adaptive upper bracket.
    """

    target = float(price)
    if target < 0:
        raise ValueError("price must be non-negative")
    if maturity <= 0:
        raise ValueError("maturity must be positive")

    opt = str(option_type).strip().lower()
    df = float(discount_factor)
    intrinsic = bs_price_forward(
        forward,
        strike,
        maturity,
        0.0,
        option_type=opt,
        discount_factor=df,
    )
    if target < intrinsic:
        return 0.0

    low = 1.0e-8
    high = 1.0
    for _ in range(50):
        high_price = bs_price_forward(
            forward,
            strike,
            maturity,
            high,
            option_type=opt,
            discount_factor=df,
        )
        if high_price >= target:
            break
        high *= 2.0
    else:
        raise RuntimeError("Failed to bracket implied volatility")

    for _ in range(int(max_iter)):
        mid = 0.5 * (low + high)
        mid_price = bs_price_forward(
            forward,
            strike,
            maturity,
            mid,
            option_type=opt,
            discount_factor=df,
        )
        err = mid_price - target
        if abs(err) <= tol:
            return float(mid)
        if err > 0.0:
            high = mid
        else:
            low = mid

    return float(0.5 * (low + high))

