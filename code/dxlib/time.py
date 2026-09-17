"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 27 - Valuation Framework.

Time-axis utilities shared across simulation, valuation, and discounting.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence

import numpy as np

__all__ = ["ensure_datetime_array", "year_fractions", "time_to_maturity"]

_SECONDS_PER_DAY = 86_400.0


def _to_datetime(value: dt.date) -> dt.datetime:
    """Return ``value`` as ``datetime.datetime`` (dates become midnight)."""
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, dt.date):
        return dt.datetime.combine(value, dt.time())
    raise TypeError(
        f"Unsupported time entry {value!r} (type {type(value).__name__})"
    )


def ensure_datetime_array(times: Sequence[dt.date]) -> np.ndarray:
    """
    Convert date-like objects to a sorted NumPy array of datetimes.

    ``datetime.datetime`` instances (a subclass of ``date``) are preserved
    as-is; plain dates become midnight timestamps. The input order does not
    matter: the result is sorted in ascending order.
    """

    if not isinstance(times, Iterable):
        raise TypeError("times must be an iterable of datetime/date objects")

    normalized = [_to_datetime(value) for value in times]
    if not normalized:
        raise ValueError("times iterable is empty")

    return np.array(sorted(normalized), dtype=object)


def year_fractions(
    times: Sequence[dt.date],
    day_count: float = 365.0,
) -> np.ndarray:
    """
    Compute ACT/day_count year fractions relative to the first
    (earliest) timestamp.
    """

    if day_count <= 0:
        raise ValueError("day_count must be strictly positive")

    ordered = ensure_datetime_array(times)
    origin = ordered[0]
    return np.array(
        [
            (ts - origin).total_seconds() / _SECONDS_PER_DAY / day_count
            for ts in ordered
        ],
        dtype=float,
    )


def time_to_maturity(
    pricing_date: dt.date,
    maturity: dt.date,
    *,
    day_count: float = 365.0,
) -> float:
    """
    Convenience helper returning the ACT/day_count time-to-maturity in years.
    """

    if day_count <= 0:
        raise ValueError("day_count must be strictly positive")

    start = _to_datetime(pricing_date)
    end = _to_datetime(maturity)
    delta_days = (end - start).total_seconds() / _SECONDS_PER_DAY
    if delta_days < 0:
        raise ValueError("maturity must be on or after pricing_date")
    return float(delta_days / day_count)
