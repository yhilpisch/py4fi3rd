"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 23 - Vectorized and Event-Based Backtesting.

Companion code for the core examples in Chapter 23:

- logistic-regression baseline for daily direction of a single instrument
- walk-forward vectorized backtest with proportional transaction costs
- minimal event-based engine fed by the same out-of-sample signals

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_EOD = ROOT / "data" / "eod_data.csv"

DEFAULT_SYMBOL = "EURUSD"
DEFAULT_TRAINING_WINDOW = 2 * 252
DEFAULT_TEST_WINDOW = 21
DEFAULT_TRANSACTION_COST = 0.0001
TRADING_DAYS = 252


def load_symbol_prices(symbol: str = DEFAULT_SYMBOL) -> pd.Series:
    """Load end-of-day prices for a single symbol from the EOD dataset."""

    if not DATA_EOD.exists():
        remote = "https://hilpisch.com/eod_data.csv"
        prices = pd.read_csv(remote, parse_dates=["Date"], index_col="Date")
    else:
        prices = pd.read_csv(DATA_EOD, parse_dates=["Date"], index_col="Date")
    return prices[symbol].dropna()


def build_logistic_regression_pipeline() -> Pipeline:
    """Create the baseline logistic-regression pipeline."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(
                C=10.0,
                solver="lbfgs",
                max_iter=500,
                random_state=0,
            )),
        ],
    )


def make_features_and_labels(
    closes: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Construct features and labels for next-period direction."""

    rets = closes.pct_change()
    next_ret = closes.shift(-1) / closes - 1.0

    data = pd.DataFrame(
        {
            "r_lag1": rets.shift(1),
            "mom_5": rets.rolling(5).mean(),
            "mom_20": rets.rolling(20).mean(),
            "next_ret": next_ret,
        }
    ).dropna()

    X = data[["r_lag1", "mom_5", "mom_20"]]
    y = (data["next_ret"] > 0.0).astype(int)
    return X, y


def walk_forward_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    training_window: int = DEFAULT_TRAINING_WINDOW,
    test_window: int = DEFAULT_TEST_WINDOW,
) -> pd.DataFrame:
    """Create out-of-sample predictions with a rolling training window."""

    if training_window < 2:
        raise ValueError("training_window must be at least 2.")
    if test_window < 1:
        raise ValueError("test_window must be at least 1.")
    if len(X) <= training_window:
        raise ValueError("Not enough data for the requested training window.")

    prediction_blocks: list[pd.DataFrame] = []

    for start in range(training_window, len(X), test_window):
        stop = min(start + test_window, len(X))
        X_train = X.iloc[start - training_window:start]
        y_train = y.iloc[start - training_window:start]
        X_test = X.iloc[start:stop]
        y_test = y.iloc[start:stop]

        model = build_logistic_regression_pipeline()
        model.fit(X_train.values, y_train.values)
        proba = model.predict_proba(X_test.values)[:, 1]

        pred_class = (proba > 0.5).astype(int)
        signal = 2.0 * pred_class - 1.0

        block = pd.DataFrame(
            {
                "y_true": y_test,
                "proba_up": proba,
                "pred_class": pred_class,
                "signal": signal,
            },
            index=X_test.index,
        )
        prediction_blocks.append(block)

    return pd.concat(prediction_blocks).sort_index()


def build_forward_market_data(closes: pd.Series) -> pd.DataFrame:
    """Build close-to-next-close market data for backtesting."""

    next_timestamp = pd.Series(closes.index, index=closes.index).shift(-1)

    market = pd.DataFrame(
        {
            "close": closes,
            "next_close": closes.shift(-1),
            "next_timestamp": next_timestamp,
        }
    ).dropna()

    market["forward_return"] = market["next_close"] / market["close"] - 1.0
    market["next_timestamp"] = pd.to_datetime(market["next_timestamp"])
    return market


def _run_signal_backtest(
    market: pd.DataFrame,
    signal: pd.Series,
    transaction_cost: float,
) -> pd.DataFrame:
    """Run a single-asset backtest with explicit cash and units."""

    cash = 1.0
    units = 0.0
    records: list[dict[str, float | pd.Timestamp]] = []

    for ts, row in market.iterrows():
        signal_i = float(signal.loc[ts])
        equity_before = cash + units * float(row["close"])
        if abs(signal_i) < 1e-12:
            target_units = 0.0
        elif abs(units) > 1e-12 and signal_i * units > 0.0:
            target_units = units
        else:
            target_units = signal_i * equity_before / float(row["close"])

        delta_units = target_units - units
        trade_value = delta_units * float(row["close"])
        trade_cost = abs(trade_value) * transaction_cost

        cash -= trade_value + trade_cost
        units = target_units
        equity_after = cash + units * float(row["next_close"])

        records.append(
            {
                "signal": signal_i,
                "position": (
                    1.0 if units > 1e-12 else -1.0 if units < -1e-12 else 0.0
                ),
                "trade_notional": abs(trade_value),
                "turnover": abs(trade_value) / equity_before,
                "return_gross": signal_i * float(row["forward_return"]),
                "return_net": equity_after / equity_before - 1.0,
                "equity": equity_after,
            }
        )

    return pd.DataFrame(records, index=market.index)


def vectorized_backtest(
    closes: pd.Series,
    predictions: pd.DataFrame,
    transaction_cost: float = DEFAULT_TRANSACTION_COST,
) -> pd.DataFrame:
    """Run a walk-forward vectorized backtest with proportional costs."""

    market = build_forward_market_data(closes)
    data = market.join(predictions, how="inner").copy()
    data["signal"] = data["signal"].astype(float)

    strat = _run_signal_backtest(
        market=data[["close", "next_close", "forward_return"]],
        signal=data["signal"],
        transaction_cost=transaction_cost,
    ).add_prefix("strat_")

    bh = _run_signal_backtest(
        market=data[["close", "next_close", "forward_return"]],
        signal=pd.Series(1.0, index=data.index),
        transaction_cost=transaction_cost,
    ).add_prefix("bh_")

    df = pd.concat([data, strat, bh], axis=1)
    df["signal_date"] = df.index
    df.index = pd.DatetimeIndex(df["next_timestamp"], name="Date")
    return df


def drawdown_diagnostics(equity: pd.Series) -> Tuple[pd.Series, int]:
    """Compute the drawdown path and the longest drawdown period."""

    drawdown = equity / equity.cummax() - 1.0
    underwater = drawdown < 0.0
    groups = (underwater != underwater.shift(fill_value=False)).cumsum()
    dd_lengths = underwater.groupby(groups).sum()
    longest_dd = int(dd_lengths.max()) if not dd_lengths.empty else 0
    return drawdown, longest_dd


def performance_summary(
    return_gross: pd.Series,
    return_net: pd.Series,
    equity: pd.Series,
    turnover: pd.Series,
) -> pd.Series:
    """Summarize strategy performance with common backtest metrics."""

    drawdown, longest_dd = drawdown_diagnostics(equity)
    ann_return = equity.iloc[-1] ** (TRADING_DAYS / len(return_net)) - 1.0
    ann_vol = return_net.std(ddof=0) * (TRADING_DAYS ** 0.5)
    downside = return_net[return_net < 0.0]
    downside_std = downside.std(ddof=0)
    downside_vol = downside_std * (TRADING_DAYS ** 0.5)

    if return_net.std(ddof=0) > 0.0:
        sharpe = (
            return_net.mean() / return_net.std(ddof=0)
            * (TRADING_DAYS ** 0.5)
        )
    else:
        sharpe = float("nan")

    if pd.notna(downside_std) and downside_std > 0.0:
        sortino = (
            return_net.mean() / downside_std
            * (TRADING_DAYS ** 0.5)
        )
    else:
        sortino = float("nan")

    if drawdown.min() < 0.0:
        calmar = ann_return / abs(float(drawdown.min()))
    else:
        calmar = float("nan")

    return pd.Series(
        {
            "gross_return": (1.0 + return_gross).prod() - 1.0,
            "net_return": equity.iloc[-1] - 1.0,
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "max_drawdown": drawdown.min(),
            "longest_dd_bars": longest_dd,
            "hit_rate": (return_net > 0.0).mean(),
            "avg_turnover": turnover.mean(),
            "cum_turnover": turnover.sum(),
            "downside_vol": downside_vol,
            "Calmar": calmar,
        }
    )


def performance_table(df_vec: pd.DataFrame) -> pd.DataFrame:
    """Create a side-by-side performance table for both strategy legs."""

    perf = pd.concat(
        {
            "buy_and_hold": performance_summary(
                return_gross=df_vec["bh_return_gross"],
                return_net=df_vec["bh_return_net"],
                equity=df_vec["bh_equity"],
                turnover=df_vec["bh_turnover"],
            ),
            "strategy": performance_summary(
                return_gross=df_vec["strat_return_gross"],
                return_net=df_vec["strat_return_net"],
                equity=df_vec["strat_equity"],
                turnover=df_vec["strat_turnover"],
            ),
        },
        axis=1,
    )
    return perf


@dataclass
class Bar:
    """Single-market bar for the event-based engine."""

    timestamp: pd.Timestamp
    close: float
    next_close: float
    signal: float


@dataclass
class Account:
    """Minimal account state for a single-asset strategy."""

    cash: float
    position: float

    def equity(self, price: float) -> float:
        return self.cash + self.position * price


@dataclass
class Order:
    """Market order for a target position in units."""

    target_position: float


class EventBacktester:
    """Minimal event-based engine with proportional transaction costs."""

    def __init__(
        self,
        starting_cash: float = 1.0,
        transaction_cost: float = DEFAULT_TRANSACTION_COST,
    ) -> None:
        self.account = Account(cash=starting_cash, position=0.0)
        self.transaction_cost = transaction_cost
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []

    def generate_order(self, bar: Bar) -> Order:
        """Translate a long-short signal into a target position in units."""

        if abs(bar.signal) < 1e-12:
            target_units = 0.0
        elif (
            abs(self.account.position) > 1e-12
            and bar.signal * self.account.position > 0.0
        ):
            target_units = self.account.position
        else:
            equity = self.account.equity(price=bar.close)
            target_units = float(bar.signal) * (equity / bar.close)

        return Order(target_position=target_units)

    def execute_order(self, bar: Bar, order: Order) -> None:
        """Execute a market order at the bar's close."""

        delta = order.target_position - self.account.position
        if abs(delta) < 1e-12:
            return
        trade_value = float(delta) * bar.close
        trade_cost = abs(trade_value) * self.transaction_cost
        self.account.cash -= trade_value + trade_cost
        self.account.position = order.target_position

    def on_bar(self, bar: Bar) -> None:
        """Process one bar: trade at the close, then mark to next close."""

        order = self.generate_order(bar)
        self.execute_order(bar, order)
        equity_next = self.account.equity(price=bar.next_close)
        self.equity_curve.append((bar.timestamp, equity_next))

    def run(self, bars: Iterable[Bar]) -> pd.Series:
        """Run the event-based backtest over a sequence of bars."""

        for bar in bars:
            self.on_bar(bar)

        index = [ts for ts, _ in self.equity_curve]
        values = [eq for _, eq in self.equity_curve]
        return pd.Series(values, index=index, name="equity_event")


def build_bars_for_strategy(df_vec: pd.DataFrame) -> Iterable[Bar]:
    """Create bars from the vectorized walk-forward backtest table."""

    for ts, row in df_vec.iterrows():
        yield Bar(
            timestamp=ts,
            close=float(row["close"]),
            next_close=float(row["next_close"]),
            signal=float(row["signal"]),
        )


def main() -> None:
    """Run a compact Chapter 23 demo and print key results."""

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

    bars = build_bars_for_strategy(df_vec)
    engine = EventBacktester(
        starting_cash=1.0,
        transaction_cost=DEFAULT_TRANSACTION_COST,
    )
    equity_event = engine.run(bars)

    aligned_eq = pd.concat(
        {
            "equity_vec": df_vec["strat_equity"],
            "equity_event": equity_event,
        },
        axis=1,
    ).dropna()
    perf = performance_table(df_vec)

    max_abs_diff = (
        aligned_eq["equity_vec"] - aligned_eq["equity_event"]
    ).abs().max()

    assert not df_vec.empty
    assert max_abs_diff < 1e-10
    assert not perf.empty

    print("== Walk-forward settings ==")
    print(
        f"training_window={DEFAULT_TRAINING_WINDOW}, "
        f"test_window={DEFAULT_TEST_WINDOW}, "
        f"transaction_cost={DEFAULT_TRANSACTION_COST:.4f}"
    )
    print()

    print("== Vectorized backtest tail ==")
    print(df_vec[["bh_equity", "strat_equity"]].tail())
    print()

    print("== Event vs. vectorized equity (tail) ==")
    print(aligned_eq.tail())
    print()
    print("== Performance summary ==")
    print(perf.round(4))
    print()
    print(f"max_abs_diff={max_abs_diff:.12f}")


if __name__ == "__main__":
    main()
