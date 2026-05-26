"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 22 - Efficient Markets and Hypothesis Testing.

Companion helpers for return-predictability diagnostics:
autocorrelation, regression-based predictability tests,
and Granger causality for simple signals.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../new
DATA_PATH = PROJECT_ROOT / "data" / "eod_data.csv"
SYMBOL = "SPY"
MAX_LAGS = 10


def load_returns(symbol: str = SYMBOL) -> pd.Series:
    """Load daily returns for a single symbol from the EOD dataset."""

    prices = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date")
    rets = prices[symbol].pct_change().dropna()
    rets.name = symbol
    return rets


def autocorrelation_table(
    returns: pd.Series,
    max_lags: int = MAX_LAGS,
) -> pd.DataFrame:
    """Return a table of sample autocorrelations with approximate std errors."""

    n = len(returns)
    lags = np.arange(1, max_lags + 1, dtype=int)
    acf_vals = [float(returns.autocorr(lag=int(k))) for k in lags]
    se = 1.0 / np.sqrt(float(n))

    return pd.DataFrame(
        {
            "lag": lags,
            "autocorrelation": acf_vals,
            "std_error": np.full(max_lags, se),
            "t_stat": np.array(acf_vals) / se,
        }
    ).set_index("lag")


def regression_predictability(
    returns: pd.Series,
    max_lag: int = 5,
) -> pd.DataFrame:
    """OLS regressions of returns on lagged returns, one per lag."""

    import statsmodels.api as sm

    results: list[dict[str, float]] = []
    rets = returns.dropna().astype(float)
    for lag in range(1, max_lag + 1):
        y = rets.iloc[lag:].values
        x_lag = rets.shift(lag).iloc[lag:].values
        X = sm.add_constant(x_lag)
        model = sm.OLS(y, X).fit()
        results.append(
            {
                "lag": lag,
                "coef_lagged": float(model.params[1]),
                "p_value": float(model.pvalues[1]),
                "r_squared": float(model.rsquared),
            }
        )

    return pd.DataFrame(results).set_index("lag")


def granger_pvalues(
    returns: pd.Series,
    window: int = 20,
    max_lag: int = 5,
) -> pd.DataFrame:
    """Granger-causality p-values for a simple momentum signal."""

    from statsmodels.tsa.stattools import grangercausalitytests

    momentum = returns.rolling(window, min_periods=window).mean()
    data = pd.concat({"r": returns, f"mom_{window}d": momentum}, axis=1).dropna()

    # Suppress printed test summaries.
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tests = grangercausalitytests(data, maxlag=max_lag, verbose=False)

    lags = np.arange(1, max_lag + 1, dtype=int)
    pvals = np.array(
        [tests[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)],
        dtype=float,
    )

    return pd.DataFrame({"lag": lags, "p_value": pvals}).set_index("lag")


def random_walk_ratio_variance(
    returns: pd.Series,
    horizons: npt.NDArray[np.int64],
) -> pd.DataFrame:
    """Variance-ratio test for the random walk null."""

    rets = returns.dropna().astype(float).values
    n = rets.size
    var_q = []
    for q in horizons:
        q_int = int(q)
        if q_int < 2 or q_int > n:
            var_q.append(np.nan)
            continue
        cum_rets = np.add.reduceat(rets, np.arange(0, n, q_int))
        var_cum = float(np.var(cum_rets, ddof=1))
        var_q.append(var_cum / (q_int * float(np.var(rets, ddof=1))))
    return pd.DataFrame({"horizon": horizons, "variance_ratio": var_q})


def main() -> None:
    """Run compact return-predictability diagnostics."""

    returns = load_returns(SYMBOL)
    acf = autocorrelation_table(returns)
    regression = regression_predictability(returns)
    granger = granger_pvalues(returns)

    print("== Autocorrelation table ==")
    print(acf.round(4))
    print()

    print("== Lagged-return regression ==")
    print(regression.round(4))
    print()

    print("== Granger-causality p-values ==")
    print(granger.round(4))
    print()

    horizons = np.array([2, 5, 10, 20], dtype=np.int64)
    vr = random_walk_ratio_variance(returns, horizons)
    print("== Variance ratios ==")
    print(vr.round(4))

    # Self-checks.
    assert len(returns) > 0, "return series must not be empty"
    assert acf["autocorrelation"].between(-1, 1).all(), (
        "autocorrelations must be in [-1, 1]"
    )
    acf1 = float(acf.loc[1, "autocorrelation"])
    assert -1.0 <= acf1 <= 1.0, "lag-1 autocorrelation out of range"
    assert granger["p_value"].between(0, 1).all(), "p-values must be in [0, 1]"


if __name__ == "__main__":
    main()
