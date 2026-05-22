"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 26 - Algorithmic Trading in the Real World.

Companion code for the chapter 26 practical examples.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HF_DATA = ROOT / "data" / "hf_data.csv"


def implementation_shortfall_example(
    gross_alpha: float,
    transaction_costs: float,
    slippage: float,
    infrastructure: float,
) -> pd.Series:
    """Summarise how frictions reduce a gross expected return."""

    net_alpha = gross_alpha - transaction_costs - slippage - infrastructure
    retention = net_alpha / gross_alpha if gross_alpha != 0.0 else np.nan
    return pd.Series(
        {
            "gross_alpha": gross_alpha,
            "transaction_costs": transaction_costs,
            "slippage": slippage,
            "infrastructure": infrastructure,
            "net_alpha": net_alpha,
            "retention_ratio": retention,
        }
    )


def break_even_hit_rate(
    avg_gain: float,
    avg_loss: float,
    transaction_cost_per_trade: float,
) -> float:
    """Compute the break-even hit rate for one-trade expected value."""

    effective_gain = avg_gain - transaction_cost_per_trade
    effective_loss = avg_loss + transaction_cost_per_trade
    return effective_loss / (effective_gain + effective_loss)


def load_hf_data(path: Path = HF_DATA) -> pd.DataFrame:
    """Load the monthly hedge-fund comparison dataset."""

    data = pd.read_csv(path)
    data.columns = data.columns.str.lower()
    data = data.rename(columns={"hf_index": "hedge_fund"})
    return (
        data.assign(date=pd.to_datetime(data["date"]))
        .set_index("date")
        .sort_index()
    )


def annualized_return(returns: pd.Series) -> float:
    """Compute the compounded annualized return from monthly data."""

    return float((1.0 + returns).prod() ** (12.0 / len(returns)) - 1.0)


def annualized_volatility(returns: pd.Series) -> float:
    """Compute annualized volatility from monthly returns."""

    return float(returns.std(ddof=0) * np.sqrt(12.0))


def max_drawdown(returns: pd.Series) -> float:
    """Compute the maximum drawdown of a return series."""

    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def sharpe_ratio(returns: pd.Series) -> float:
    """Compute a zero-rate Sharpe ratio from monthly returns."""

    ann_vol = annualized_volatility(returns)
    if ann_vol == 0.0:
        return float("nan")
    return annualized_return(returns) / ann_vol


def sortino_ratio(returns: pd.Series) -> float:
    """Compute a zero-rate Sortino ratio from monthly returns."""

    downside = returns[returns < 0.0].std(ddof=0) * np.sqrt(12.0)
    if pd.isna(downside) or downside == 0.0:
        return float("nan")
    return annualized_return(returns) / float(downside)


def build_hf_summary_table(data: pd.DataFrame) -> pd.DataFrame:
    """Create a compact return-risk table for the comparison assets."""

    rets = data.copy()
    rets["6040"] = 0.6 * rets["spy"] + 0.4 * rets["ief"]
    spy_vol = annualized_volatility(rets["spy"])
    hf_vol = annualized_volatility(rets["hedge_fund"])
    leverage = spy_vol / hf_vol
    rets["hf_lev"] = leverage * rets["hedge_fund"]

    rows = []
    for column in ["hedge_fund", "spy", "ief", "6040", "hf_lev"]:
        series = rets[column]
        rows.append(
            {
                "series": column,
                "ann_return": annualized_return(series),
                "ann_vol": annualized_volatility(series),
                "cum_return": float((1.0 + series).prod() - 1.0),
                "max_dd": max_drawdown(series),
                "sharpe": sharpe_ratio(series),
                "sortino": sortino_ratio(series),
            }
        )
    return pd.DataFrame(rows).set_index("series")


def benchmark_diagnostics(data: pd.DataFrame, column: str) -> pd.Series:
    """Compute benchmark-relative diagnostics versus SPY."""

    series = data[column]
    benchmark = data["spy"]
    beta = np.cov(series, benchmark, ddof=0)[0, 1] / np.var(benchmark, ddof=0)
    tracking_error = (series - benchmark).std(ddof=0) * np.sqrt(12.0)
    active_ann_return = annualized_return(series) - annualized_return(benchmark)
    info_ratio = (
        active_ann_return / tracking_error if tracking_error > 0.0 else np.nan
    )
    up_mask = benchmark > 0.0
    down_mask = benchmark < 0.0
    return pd.Series(
        {
            "beta": float(beta),
            "corr": float(series.corr(benchmark)),
            "tracking_error": float(tracking_error),
            "active_ann_return": float(active_ann_return),
            "information_ratio": float(info_ratio),
            "up_capture": float(
                series[up_mask].mean() / benchmark[up_mask].mean()
            ),
            "down_capture": float(
                series[down_mask].mean() / benchmark[down_mask].mean()
            ),
        }
    )


def calendar_return_table(data: pd.DataFrame) -> pd.DataFrame:
    """Compute calendar-year returns for the main comparison series."""

    rets = data.copy()
    rets["6040"] = 0.6 * rets["spy"] + 0.4 * rets["ief"]
    leverage = annualized_volatility(rets["spy"]) / annualized_volatility(
        rets["hedge_fund"]
    )
    rets["hf_lev"] = leverage * rets["hedge_fund"]
    return (
        (1.0 + rets[["hedge_fund", "spy", "6040", "hf_lev"]])
        .groupby(rets.index.year)
        .prod()
        - 1.0
    )


def main() -> None:
    shortfall = implementation_shortfall_example(
        gross_alpha=0.08,
        transaction_costs=0.015,
        slippage=0.010,
        infrastructure=0.005,
    )
    hit_rate = break_even_hit_rate(
        avg_gain=0.012,
        avg_loss=0.010,
        transaction_cost_per_trade=0.001,
    )

    hf_data = load_hf_data()
    summary = build_hf_summary_table(hf_data)
    diagnostics_hf = benchmark_diagnostics(hf_data, "hedge_fund")
    diagnostics_6040 = benchmark_diagnostics(
        hf_data.assign(**{"6040": 0.6 * hf_data["spy"] + 0.4 * hf_data["ief"]}),
        "6040",
    )
    diagnostics_hf_lev = benchmark_diagnostics(
        hf_data.assign(
            **{
                "hf_lev": (
                    annualized_volatility(hf_data["spy"])
                    / annualized_volatility(hf_data["hedge_fund"])
                ) * hf_data["hedge_fund"]
            }
        ),
        "hf_lev",
    )
    calendar = calendar_return_table(hf_data)

    assert shortfall["net_alpha"] < shortfall["gross_alpha"]
    assert 0.0 < hit_rate < 1.0
    assert not summary.empty

    print("SECTION: shortfall")
    print(shortfall.round(4))
    print(round(hit_rate, 4))

    print("\nSECTION: summary")
    print(summary.round(4))

    print("\nSECTION: diagnostics_hf")
    print(diagnostics_hf.round(4))

    print("\nSECTION: diagnostics_6040")
    print(diagnostics_6040.round(4))

    print("\nSECTION: diagnostics_hf_lev")
    print(diagnostics_hf_lev.round(4))

    print("\nSECTION: calendar")
    print(calendar.round(4))


if __name__ == "__main__":
    main()
