"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 24 - Building a Market and Broker for Trading.

Compact trading engine package for chapter 24.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from engine.broker import PaperBroker
    from engine.data import (
        GBMFeed,
        HistoricalFeed,
        build_feed,
        estimate_gbm_parameters,
        generate_simulated_timestamps,
    )
    from engine.models import OrderRequest, Position, StopOrder, Tick
    from engine.session import TradingSession
else:
    from .broker import PaperBroker
    from .data import (
        GBMFeed,
        HistoricalFeed,
        build_feed,
        estimate_gbm_parameters,
        generate_simulated_timestamps,
    )
    from .models import OrderRequest, Position, StopOrder, Tick
    from .session import TradingSession

__all__ = [
    "GBMFeed",
    "HistoricalFeed",
    "OrderRequest",
    "PaperBroker",
    "Position",
    "StopOrder",
    "Tick",
    "TradingSession",
    "build_feed",
    "estimate_gbm_parameters",
    "generate_simulated_timestamps",
]
