"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 21 - A Small Asset Management Library in Python.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from assetlib.core import Instrument, Portfolio, Position, Universe
    from assetlib.data import MarketData
    from assetlib.signals import SignalEngine
else:
    from .core import Instrument, Portfolio, Position, Universe
    from .data import MarketData
    from .signals import SignalEngine

__all__ = [
    "Instrument",  # core.Instrument re-export
    "Universe",  # core.Universe re-export
    "Position",  # core.Position re-export
    "Portfolio",  # core.Portfolio re-export
    "MarketData",  # data.MarketData re-export
    "SignalEngine",  # signals.SignalEngine re-export
]
