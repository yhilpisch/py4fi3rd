"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 23 - Vectorized and Event-Based Backtesting.

Vectorized equity curves for buy-and-hold vs ML strategy.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from types import ModuleType

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "assets" / "figures" / "ch23_vec_equity_curves.png"

FIGSIZE = (6.8, 3.5)
DPI = 300
BASE_FONT_SIZE = 10
TITLE_FONT_SIZE = 11

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_ch23_module() -> ModuleType:
    """Import the chapter module in validator, script, and notebook contexts."""

    try:
        return importlib.import_module(
            "code.chapters.ch23_vectorized_and_event_based_backtesting"
        )
    except ModuleNotFoundError:
        return importlib.import_module(
            "chapters.ch23_vectorized_and_event_based_backtesting"
        )


CH23 = _load_ch23_module()
DEFAULT_SYMBOL = CH23.DEFAULT_SYMBOL
DEFAULT_TEST_WINDOW = CH23.DEFAULT_TEST_WINDOW
DEFAULT_TRAINING_WINDOW = CH23.DEFAULT_TRAINING_WINDOW
DEFAULT_TRANSACTION_COST = CH23.DEFAULT_TRANSACTION_COST
load_symbol_prices = CH23.load_symbol_prices
make_features_and_labels = CH23.make_features_and_labels
walk_forward_predictions = CH23.walk_forward_predictions
vectorized_backtest = CH23.vectorized_backtest


def main() -> None:
    """Plot equity curves for buy-and-hold vs ML strategy (vectorized)."""

    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": BASE_FONT_SIZE,
            "axes.titlesize": TITLE_FONT_SIZE,
        }
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    closes = load_symbol_prices(symbol=DEFAULT_SYMBOL)
    X, y = make_features_and_labels(closes)
    pred_df = walk_forward_predictions(
        X=X,
        y=y,
        training_window=DEFAULT_TRAINING_WINDOW,
        test_window=DEFAULT_TEST_WINDOW,
    )
    df_vec = vectorized_backtest(
        closes=closes,
        predictions=pred_df,
        transaction_cost=DEFAULT_TRANSACTION_COST,
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(
        df_vec.index,
        df_vec["bh_equity"],
        color="C0",
        linewidth=1.0,
        label="Buy-and-hold",
    )
    ax.plot(
        df_vec.index,
        df_vec["strat_equity"],
        color="C1",
        linewidth=1.0,
        label="ML strategy",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalised equity (start = 1.0)")
    ax.set_title("Walk-Forward Backtest with Proportional Costs")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=DPI)


if __name__ == "__main__":
    main()
