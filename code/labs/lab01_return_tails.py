"""Python for Finance, 3rd ed., O'Reilly (2026).
Lab 01 - The Importance of Return Tails for Investing and Trading.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "eod_data.csv"


def load_prices() -> pd.DataFrame:
    """Load the local daily price data used across the lab."""
    return pd.read_csv(DATA_FILE, parse_dates=["Date"], index_col="Date")


def load_returns(symbol: str = "SPY") -> pd.Series:
    """Return simple daily returns for one symbol."""
    prices = load_prices()
    rets = prices[symbol].pct_change().dropna()
    rets.name = symbol
    return rets


def tail_summary(returns: pd.Series, n: int = 10) -> pd.DataFrame:
    """Summarize the largest and smallest daily returns."""
    ordered = returns.sort_values()
    worst = ordered.head(n)
    best = ordered.tail(n)
    frame = pd.DataFrame(
        {
            "worst": worst.values,
            "best": best.sort_values(ascending=False).values,
        }
    )
    return frame


def cumulative_path(returns: pd.Series) -> pd.Series:
    """Compound simple returns into a normalized cumulative path."""
    return (1.0 + returns).cumprod()


def remove_extreme_days(
    returns: pd.Series, n: int = 10, side: str = "best"
) -> pd.Series:
    """Replace the largest or smallest daily returns by zero."""
    out = returns.copy()
    if side == "best":
        idx = out.nlargest(n).index
    elif side == "worst":
        idx = out.nsmallest(n).index
    else:
        raise ValueError("side must be 'best' or 'worst'")
    out.loc[idx] = 0.0
    return out


def contribution_summary(returns: pd.Series, n: int = 10) -> pd.DataFrame:
    """Show how best and worst days contribute to the full sample."""
    full = cumulative_path(returns).iloc[-1] - 1.0
    no_best = cumulative_path(remove_extreme_days(returns, n=n, side="best"))
    no_worst = cumulative_path(remove_extreme_days(returns, n=n, side="worst"))
    data = {
        "scenario": ["full_sample", "without_best_days", "without_worst_days"],
        "terminal_return": [
            full,
            no_best.iloc[-1] - 1.0,
            no_worst.iloc[-1] - 1.0,
        ],
    }
    return pd.DataFrame(data)


def drawdown(returns: pd.Series) -> pd.Series:
    """Compute drawdown from a cumulative path."""
    wealth = cumulative_path(returns)
    peak = wealth.cummax()
    return wealth / peak - 1.0


def rolling_tail_share(returns: pd.Series, window: int = 252) -> pd.Series:
    """Measure how much of total absolute move comes from the largest 5 days."""
    values = []
    index = []
    abs_rets = returns.abs()
    for end in range(window, len(abs_rets) + 1):
        sample = abs_rets.iloc[end - window : end]
        share = sample.nlargest(5).sum() / sample.sum()
        values.append(float(share))
        index.append(sample.index[-1])
    return pd.Series(values, index=index, name="tail_share")


def main() -> None:
    """Print a few compact tail summaries for manual inspection."""
    rets = load_returns("SPY")
    summary = contribution_summary(rets, n=10)
    print(summary.to_string(index=False))
    print()
    print(tail_summary(rets, n=5).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
