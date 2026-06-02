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
from typing import overload

import numpy as np

__all__ = ["ensure_datetime_array", "year_fractions", "time_to_maturity"]


@overload
def ensure_datetime_array(times: Sequence[dt.datetime]) -> np.ndarray: ...


@overload
def ensure_datetime_array(times: Sequence[dt.date]) -> np.ndarray: ...


def ensure_datetime_array(
    times: Sequence[dt.date | dt.datetime],
) -> np.ndarray:
    """
    Convert ``datetime``/``date`` objects to a sorted NumPy array.

    The array uses dtype ``object`` and contains ``datetime.datetime`` objects.
    """

    if not isinstance(times, Iterable):
        raise TypeError("times must be an iterable of datetime/date objects")

    normalized: list[dt.datetime] = []
    for value in times:
        if isinstance(value, dt.datetime):
            normalized.append(value)
        elif isinstance(value, dt.date):
            normalized.append(dt.datetime.combine(value, dt.time()))
        else:  # pragma: no cover
            msg = (
                "Unsupported time entry "
                f"{value!r} (type {type(value).__name__})"
            )
            raise TypeError(msg)

    if not normalized:
        raise ValueError("times iterable is empty")

    return np.array(sorted(normalized), dtype=object)


def year_fractions(
    times: Sequence[dt.date | dt.datetime],
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
    deltas: list[float] = []
    for ts in ordered:
        days = (ts - origin).total_seconds() / 86_400.0
        deltas.append(days / day_count)
    return np.array(deltas, dtype=float)


def time_to_maturity(
    pricing_date: dt.date | dt.datetime,
    maturity: dt.date | dt.datetime,
    *,
    day_count: float = 365.0,
) -> float:
    """
    Convenience helper returning the ACT/day_count time-to-maturity in years.
    """

    if day_count <= 0:
        raise ValueError("day_count must be strictly positive")

    dates = []
    for value in (pricing_date, maturity):
        if isinstance(value, dt.datetime):
            dates.append(value)
        elif isinstance(value, dt.date):
            dates.append(dt.datetime.combine(value, dt.time()))
        else:  # pragma: no cover
            msg = (
                "Unsupported time entry "
                f"{value!r} (type {type(value).__name__})"
            )
            raise TypeError(msg)

    delta_days = (dates[1] - dates[0]).total_seconds() / 86_400.0
    if delta_days < 0:
        raise ValueError("maturity must be on or after pricing_date")
    return float(delta_days / day_count)
