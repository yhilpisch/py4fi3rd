"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 2 - Python Fluency and GenAI.

This module provides slightly more production-ready utilities that mirror
Chapter 2's themes:

- small, testable functions for numerical work,
- explicit type hints and docstrings,
- and simple "validation hooks" that you can reuse when collaborating
  with a GenAI assistant.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def gross_and_net(
    values: Iterable[float],
    fee_rate: float,
) -> tuple[float, float]:
    """Return gross and net after applying a proportional fee.

    Parameters
    ----------
    values:
        Iterable of cash flows or trade notionals.
    fee_rate:
        Proportional fee (e.g. ``0.0015`` for 15 bps).
    """

    arr = np.asarray(list(values), dtype=float)
    gross = float(arr.sum())
    fees = float(fee_rate * gross)
    net = gross - fees
    return gross, net


@dataclass
class ValidationResult:
    """Container for simple validation outcomes."""

    ok: bool
    message: str


def validate_proportions(
    values: Iterable[float],
    rtol: float = 1e-6,
) -> ValidationResult:
    """Validate that proportions sum to one within a tolerance."""

    arr = np.asarray(list(values), dtype=float)
    total = float(arr.sum())
    if not np.isclose(total, 1.0, rtol=rtol):
        return ValidationResult(
            ok=False,
            message=f"proportions sum to {total:.6f}, expected 1.0",
        )
    return ValidationResult(
        ok=True,
        message="proportions sum to one within tolerance",
    )


def main() -> None:
    """Run a short self-check demo."""

    gross, net = gross_and_net([100.0, -40.0, 60.0], fee_rate=0.001)
    assert np.isclose(gross, 120.0)
    assert net < gross

    result = validate_proportions([0.6, 0.3, 0.1])
    assert result.ok


if __name__ == "__main__":
    main()
