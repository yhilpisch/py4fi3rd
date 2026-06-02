"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 30 - Portfolio Valuation.

Portfolio containers and simple bump-and-revalue sensitivities.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = [
    "Pricer",
    "Position",
    "Portfolio",
    "price_and_stderr",
    "delta_central",
]


class Pricer(Protocol):
    """
    Minimal pricing interface used by `dxlib.portfolio`.

    The `value()` method may return either:

    - a `(price, stderr)` tuple, or
    - a dictionary with keys `"price"` and `"stderr"`.
    """

    def value(self, spot: float) -> Any: ...


def price_and_stderr(pricer: Pricer, spot: float) -> tuple[float, float]:
    """
    Normalize different pricer return types to `(price, stderr)`.
    """

    raw = pricer.value(spot)
    if isinstance(raw, tuple) and len(raw) == 2:
        price, stderr = raw
        return float(price), float(stderr)
    if isinstance(raw, dict):
        price = raw.get("price")
        stderr = raw.get("stderr")
        if price is None or stderr is None:
            raise ValueError(
                "dict result must contain keys 'price' and 'stderr'"
            )
        return float(price), float(stderr)
    raise TypeError("Unsupported pricer return type")


def delta_central(
    pricer: Pricer,
    spot: float,
    *,
    rel_bump: float = 0.01,
) -> float:
    """
    Central-difference delta estimate via bump-and-revalue.
    """

    if spot <= 0:
        raise ValueError("spot must be positive")
    if rel_bump <= 0:
        raise ValueError("rel_bump must be positive")

    bump = float(spot) * float(rel_bump)
    up, _ = price_and_stderr(pricer, spot + bump)
    down, _ = price_and_stderr(pricer, spot - bump)
    return float((up - down) / (2.0 * bump))


@dataclass(frozen=True, slots=True)
class Position:
    """
    A portfolio position defined by a quantity and a pricer.
    """

    name: str
    quantity: float
    pricer: Pricer

    def value(self, spot: float) -> tuple[float, float]:
        price, stderr = price_and_stderr(self.pricer, spot)
        return float(self.quantity) * price, abs(float(self.quantity)) * stderr

    def delta(self, spot: float, *, rel_bump: float = 0.01) -> float:
        return float(self.quantity) * delta_central(
            self.pricer,
            spot,
            rel_bump=rel_bump,
        )


@dataclass(frozen=True, slots=True)
class Portfolio:
    """
    A collection of positions with simple aggregation logic.
    """

    name: str
    positions: tuple[Position, ...]

    def value(self, spot: float) -> dict[str, object]:
        values: list[dict[str, float]] = []
        total_value = 0.0
        total_var = 0.0
        for pos in self.positions:
            pos_value, pos_stderr = pos.value(spot)
            total_value += pos_value
            total_var += pos_stderr**2
            values.append(
                {
                    "name": pos.name,
                    "value": float(pos_value),
                    "stderr": float(pos_stderr),
                    "quantity": float(pos.quantity),
                }
            )

        total_stderr = float(math.sqrt(total_var))
        return {
            "name": self.name,
            "spot": float(spot),
            "positions": values,
            "total_value": float(total_value),
            "total_stderr": total_stderr,
            "stderr_method": "independent_root_sum_squares",
        }

    def deltas(
        self,
        spot: float,
        *,
        rel_bump: float = 0.01,
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for pos in self.positions:
            out[pos.name] = pos.delta(spot, rel_bump=rel_bump)
        out["portfolio"] = float(sum(out.values()))
        return out
