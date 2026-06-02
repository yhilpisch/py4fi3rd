"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 27 - Valuation Framework.

Market environment container.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import datetime as dt
import sys
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    package_dir = Path(__file__).resolve().parent
    sys.path = [
        path
        for path in sys.path
        if Path(path or ".").resolve() != package_dir
    ]
    sys.path.insert(0, str(package_dir.parent))
    __package__ = "dxlib"

from .curves import DiscountCurve

__all__ = ["MarketEnvironment"]


@dataclass
class MarketEnvironment:
    """
    Container bundling pricing date, constants, lists, and curves.

    The design mirrors what many derivatives libraries do in practice:
    instead of
    passing dozens of arguments to every pricer, you pass a single environment
    object and extract what you need.
    """

    name: str
    pricing_date: dt.date | dt.datetime
    constants: MutableMapping[str, float] = field(default_factory=dict)
    lists: MutableMapping[str, list[str]] = field(default_factory=dict)
    curves: MutableMapping[str, DiscountCurve] = field(default_factory=dict)

    def add_constant(self, key: str, value: float) -> None:
        self.constants[key] = float(value)

    def get_constant(self, key: str) -> float:
        return float(self.constants[key])

    def add_list(self, key: str, values: Iterable[str]) -> None:
        self.lists[key] = list(values)

    def get_list(self, key: str) -> list[str]:
        return list(self.lists[key])

    def add_curve(self, key: str, curve: DiscountCurve) -> None:
        self.curves[key] = curve

    def get_curve(self, key: str) -> DiscountCurve:
        return self.curves[key]

    def merge(self, other: MarketEnvironment) -> None:
        """
        Merge another environment into this one, overriding duplicate keys.
        """

        self.constants.update(other.constants)
        self.lists.update(other.lists)
        self.curves.update(other.curves)

    def snapshot(self) -> Mapping[str, Mapping[str, object]]:
        """
        Return an immutable snapshot for logging/debugging.
        """

        return {
            "constants": dict(self.constants),
            "lists": {key: list(values) for key, values in self.lists.items()},
            "curves": dict(self.curves),
        }
