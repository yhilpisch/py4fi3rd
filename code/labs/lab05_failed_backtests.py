"""Python for Finance, 3rd ed., O'Reilly (2026).
Lab 05 - Why Failed Backtests Still Look Convincing.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "eod_data.csv"
SYMBOL = "SPY"
TRADING_DAYS = 252


def load_returns(symbol: str = SYMBOL) -> pd.Series:
    """Load simple daily returns for one symbol from the local dataset."""
    prices = pd.read_csv(DATA_FILE, parse_dates=["Date"], index_col="Date")
    rets = prices[symbol].pct_change().dropna()
    rets.name = symbol
    return rets


def equity_curve(returns: pd.Series) -> pd.Series:
    """Compound simple returns into a normalized equity curve."""
    return (1.0 + returns.fillna(0.0)).cumprod()


def annualized_sharpe(returns: pd.Series) -> float:
    """Compute an annualized Sharpe ratio without a risk-free adjustment."""
    clean = returns.dropna()
    if clean.std() == 0.0:
        return np.nan
    return float(clean.mean() / clean.std() * np.sqrt(TRADING_DAYS))


def summarize_returns(returns: pd.Series) -> dict[str, float]:
    """Summarize terminal return, volatility, drawdown, and Sharpe ratio."""
    curve = equity_curve(returns)
    drawdown = curve / curve.cummax() - 1.0
    return {
        "terminal_return": float(curve.iloc[-1] - 1.0),
        "volatility": float(returns.std() * np.sqrt(TRADING_DAYS)),
        "max_drawdown": float(drawdown.min()),
        "sharpe": annualized_sharpe(returns),
    }


def lookahead_comparison(symbol: str = SYMBOL) -> pd.DataFrame:
    """Compare buy-and-hold, lagged sign, and impossible same-day sign."""
    rets = load_returns(symbol)
    lagged_position = np.sign(rets.shift(1)).fillna(0.0)
    same_day_position = np.sign(rets)
    strategies = {
        "Buy and hold": rets,
        "Lagged sign": lagged_position * rets,
        "Same-day sign": same_day_position * rets,
    }
    return pd.DataFrame(strategies).dropna()


def momentum_returns(
    lookback: int = 20,
    cost: float = 0.0005,
    symbol: str = SYMBOL,
) -> pd.DataFrame:
    """Build a lagged momentum strategy with gross and net returns."""
    rets = load_returns(symbol)
    raw_signal = np.sign(rets.rolling(lookback).mean())
    position = raw_signal.shift(1).fillna(0.0)
    gross = position * rets
    turnover = position.diff().abs().fillna(position.abs())
    net = gross - turnover * cost
    return pd.DataFrame(
        {
            "position": position,
            "gross": gross,
            "net": net,
            "turnover": turnover,
        }
    ).dropna()


def cost_comparison(lookback: int = 20) -> pd.DataFrame:
    """Return summary statistics for gross and cost-aware momentum returns."""
    data = momentum_returns(lookback=lookback)
    rows = []
    for name in ["gross", "net"]:
        stats = summarize_returns(data[name])
        stats["strategy"] = name
        rows.append(stats)
    return pd.DataFrame(rows).set_index("strategy")


def parameter_search(
    lookbacks: tuple[int, ...] = (5, 10, 20, 40, 80, 120),
    split_date: str = "2023-01-01",
) -> pd.DataFrame:
    """Compare in-sample and out-of-sample Sharpe ratios by lookback."""
    rows = []
    for lookback in lookbacks:
        data = momentum_returns(lookback=lookback)
        train = data.loc[data.index < split_date, "net"]
        test = data.loc[data.index >= split_date, "net"]
        rows.append(
            {
                "lookback": lookback,
                "train_sharpe": annualized_sharpe(train),
                "test_sharpe": annualized_sharpe(test),
                "train_terminal": equity_curve(train).iloc[-1] - 1.0,
                "test_terminal": equity_curve(test).iloc[-1] - 1.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Print compact summaries for manual inspection."""
    lookahead = lookahead_comparison()
    lookahead_summary = pd.DataFrame(
        {
            name: summarize_returns(lookahead[name])
            for name in lookahead.columns
        }
    ).T
    print("LOOK-AHEAD COMPARISON")
    print(lookahead_summary.round(3).to_string())
    print()

    print("COST COMPARISON")
    print(cost_comparison().round(3).to_string())
    print()

    print("PARAMETER SEARCH")
    print(parameter_search().round(3).to_string(index=False))


if __name__ == "__main__":
    main()
