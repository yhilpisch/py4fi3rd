"""Python for Finance, 3rd ed., O'Reilly (2026).
Lab 06 - Alpha Is Rare.

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
DATA_FILE = ROOT / "data" / "hf_data.csv"
MONTHS_PER_YEAR = 12


def load_returns() -> pd.DataFrame:
    """Load the aligned monthly hedge fund, SPY, and IEF return sample."""
    data = pd.read_csv(DATA_FILE, parse_dates=["DATE"])
    cols = ["DATE", "HF_INDEX", "SPY", "IEF"]
    missing = sorted(set(cols) - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return data[cols].set_index("DATE").sort_index()


def cumulative_wealth(returns: pd.DataFrame) -> pd.DataFrame:
    """Compound monthly returns into cumulative wealth paths."""
    return (1.0 + returns).cumprod()


def annualized_return(returns: pd.Series) -> float:
    """Compute geometric annualized return from monthly returns."""
    growth = float((1.0 + returns).prod())
    years = len(returns) / MONTHS_PER_YEAR
    return growth ** (1.0 / years) - 1.0


def annualized_volatility(returns: pd.Series) -> float:
    """Compute annualized volatility from monthly returns."""
    return float(returns.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))


def sharpe_ratio(returns: pd.Series) -> float:
    """Compute the annualized Sharpe ratio with zero risk-free rate."""
    vol = annualized_volatility(returns)
    if vol == 0.0:
        return np.nan
    return float(returns.mean() * MONTHS_PER_YEAR / vol)


def max_drawdown(returns: pd.Series) -> float:
    """Compute maximum drawdown from a return stream."""
    wealth = (1.0 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def capture_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    benchmark_positive: bool,
) -> float:
    """Compute compounded up- or down-capture versus the benchmark."""
    if benchmark_positive:
        mask = benchmark_returns > 0.0
    else:
        mask = benchmark_returns < 0.0
    strategy = strategy_returns.loc[mask]
    benchmark = benchmark_returns.loc[mask]
    if strategy.empty or benchmark.empty:
        return np.nan
    strategy_growth = float((1.0 + strategy).prod() - 1.0)
    benchmark_growth = float((1.0 + benchmark).prod() - 1.0)
    if benchmark_growth == 0.0:
        return np.nan
    return strategy_growth / benchmark_growth


def comparison_summary(returns: pd.DataFrame) -> pd.DataFrame:
    """Summarize the hedge fund index and SPY side by side."""
    rows = []
    for name, col in [("Hedge fund index", "HF_INDEX"), ("SPY", "SPY")]:
        series = returns[col]
        rows.append(
            {
                "portfolio": name,
                "total_return": float((1.0 + series).prod() - 1.0),
                "annualized_return": annualized_return(series),
                "annualized_vol": annualized_volatility(series),
                "sharpe": sharpe_ratio(series),
                "max_drawdown": max_drawdown(series),
            }
        )
    return pd.DataFrame(rows)


def alpha_summary(returns: pd.DataFrame) -> pd.Series:
    """Estimate a simple single-factor alpha against SPY."""
    hf_index = returns["HF_INDEX"]
    benchmark = returns["SPY"]
    active = hf_index - benchmark
    variance = benchmark.var(ddof=1)
    beta = hf_index.cov(benchmark) / variance if variance != 0.0 else np.nan
    monthly_alpha = hf_index.mean() - beta * benchmark.mean()
    tracking_error = active.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)
    active_annualized = (1.0 + active.mean()) ** MONTHS_PER_YEAR - 1.0
    return pd.Series(
        {
            "months": len(returns),
            "correlation": hf_index.corr(benchmark),
            "beta": beta,
            "monthly_alpha": monthly_alpha,
            "annualized_alpha": (1.0 + monthly_alpha) ** MONTHS_PER_YEAR - 1.0,
            "tracking_error": tracking_error,
            "information_ratio": active_annualized / tracking_error,
            "up_capture": capture_ratio(
                hf_index, benchmark, benchmark_positive=True
            ),
            "down_capture": capture_ratio(
                hf_index, benchmark, benchmark_positive=False
            ),
            "outperform_months": int((active > 0.0).sum()),
        }
    )


def scenario_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Build simple comparison portfolios from the monthly return sample."""
    out = pd.DataFrame(index=returns.index)
    out["HF_INDEX"] = returns["HF_INDEX"]
    out["SPY"] = returns["SPY"]
    out["60_40"] = 0.60 * returns["SPY"] + 0.40 * returns["IEF"]
    hf_vol = annualized_volatility(returns["HF_INDEX"])
    spy_vol = annualized_volatility(returns["SPY"])
    scale = spy_vol / hf_vol
    out["LEVERED_HF"] = scale * returns["HF_INDEX"]
    return out


def scenario_summary(returns: pd.DataFrame) -> pd.DataFrame:
    """Summarize base and hypothetical comparison portfolios."""
    scenarios = scenario_returns(returns)
    labels = {
        "HF_INDEX": "Hedge fund index",
        "SPY": "SPY",
        "60_40": "60/40 SPY-IEF",
        "LEVERED_HF": "Levered hedge fund",
    }
    rows = []
    for col, label in labels.items():
        series = scenarios[col]
        rows.append(
            {
                "portfolio": label,
                "total_return": float((1.0 + series).prod() - 1.0),
                "annualized_return": annualized_return(series),
                "annualized_vol": annualized_volatility(series),
                "sharpe": sharpe_ratio(series),
                "max_drawdown": max_drawdown(series),
                "beta_to_SPY": (
                    series.cov(returns["SPY"]) / returns["SPY"].var()
                ),
            }
        )
    return pd.DataFrame(rows)


def yearly_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly hedge fund and SPY returns by calendar year."""
    work = returns[["HF_INDEX", "SPY"]].copy()
    work["year"] = work.index.year
    table = work.groupby("year").agg(
        HF_INDEX=("HF_INDEX", lambda x: (1.0 + x).prod() - 1.0),
        SPY=("SPY", lambda x: (1.0 + x).prod() - 1.0),
    )
    table["ACTIVE"] = table["HF_INDEX"] - table["SPY"]
    return table


def main() -> None:
    """Print compact summaries for manual inspection."""
    returns = load_returns()
    print("COMPARISON")
    print(comparison_summary(returns).round(3).to_string(index=False))
    print()
    print("ALPHA SUMMARY")
    print(alpha_summary(returns).round(3).to_string())
    print()
    print("SCENARIOS")
    print(scenario_summary(returns).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
