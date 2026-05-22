"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 25 - Automated Deployment of Trading Strategies.

Companion code for the chapter 25 deployment examples.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any
from types import ModuleType

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SYMBOL = "EURUSD"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_engine_module() -> ModuleType:
    """Import the local engine package in script and notebook contexts."""

    try:
        return importlib.import_module("code.engine")
    except ModuleNotFoundError:
        return importlib.import_module("engine")


ENGINE = _load_engine_module()
HistoricalFeed = ENGINE.HistoricalFeed
PaperBroker = ENGINE.PaperBroker
TradingSession = ENGINE.TradingSession
build_feed = ENGINE.build_feed


ROOT = Path(__file__).resolve().parents[2]
TMP_DIR = ROOT / "_tmp" / "ch25"


def build_logistic_regression_pipeline() -> Pipeline:
    """Create the baseline logistic-regression pipeline."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),  # normalize each refit window
            (
                "logreg",
                LogisticRegression(
                    C=10.0,
                    solver="lbfgs",
                    max_iter=500,
                    random_state=0,
                ),
            ),
        ],
    )


def make_features_and_labels(
    closes: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Construct lagged features and next-bar direction labels."""

    rets = closes.pct_change()  # simple bar returns
    next_ret = closes.shift(-1) / closes - 1.0  # one-bar-ahead return
    data = pd.DataFrame(
        {
            "r_lag1": rets.shift(1),
            "mom_3": rets.rolling(3).mean(),
            "mom_5": rets.rolling(5).mean(),
            "next_ret": next_ret,
        }
    ).dropna()

    X = data[["r_lag1", "mom_3", "mom_5"]]
    y = (data["next_ret"] > 0.0).astype(int)
    return X, y


def make_latest_feature_row(closes: pd.Series) -> pd.DataFrame | None:
    """Construct the latest feature row for the next prediction."""

    rets = closes.pct_change()
    row = pd.DataFrame(
        {
            "r_lag1": [rets.shift(1).iloc[-1]],
            "mom_3": [rets.rolling(3).mean().iloc[-1]],
            "mom_5": [rets.rolling(5).mean().iloc[-1]],
        },
        index=[closes.index[-1]],
    )
    if row.isna().any(axis=None):
        return None
    return row


@dataclass
class DeploymentConfig:
    """Configuration for the deployable baseline ML strategy."""

    sample_interval: str
    training_window: int
    retrain_interval: int
    position_size: float
    probability_threshold: float = 0.5


class DeploymentLogger:
    """Collect structured deployment events and optionally persist them."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.events: list[dict[str, Any]] = []

    def log(
        self,
        event: str,
        timestamp: pd.Timestamp,
        **payload: Any,
    ) -> None:
        self.events.append(
            {
                "logger": self.name,
                "event": event,
                "timestamp": pd.Timestamp(timestamp),
                **payload,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.events)
        if frame.empty:
            return frame
        return frame.sort_values("timestamp").reset_index(drop=True)

    def write_jsonl(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for event in self.events:
                payload = dict(event)
                payload["timestamp"] = pd.Timestamp(
                    payload["timestamp"]
                ).isoformat()
                handle.write(json.dumps(payload) + "\n")
        return path


class BaselineMLDeployer:
    """Deploy the chapter 23 baseline model on resampled bars."""

    def __init__(
        self,
        config: DeploymentConfig,
        logger: DeploymentLogger,
    ) -> None:
        self.config = config
        self.logger = logger
        self.tick_history: list[dict[str, object]] = []
        self.latest_bar_time: pd.Timestamp | None = None
        self.latest_signal: int = 0
        self.latest_probability: float | None = None
        self.model: Pipeline | None = None
        self.last_fit_bar_count = 0

    def on_tick(self, session: TradingSession, tick) -> None:
        """Update bars, refit when needed, and trade only on signal changes."""

        self.tick_history.append(
            {
                "timestamp": pd.Timestamp(tick.timestamp),
                "mid": tick.mid,
            }
        )
        bars = self._completed_bars()
        if bars.empty:
            return

        bar_time = pd.Timestamp(bars.index[-1])
        if (
            self.latest_bar_time is not None
            and bar_time <= self.latest_bar_time
        ):
            return

        self.latest_bar_time = bar_time
        self.logger.log(
            "bar",
            bar_time,
            close=float(bars.iloc[-1]),
            bars_seen=int(len(bars)),
        )

        X_trainable, y_trainable = make_features_and_labels(bars)
        if len(X_trainable) < self.config.training_window:
            self.logger.log(
                "warmup",
                bar_time,
                bars_seen=int(len(bars)),
                rows_available=int(len(X_trainable)),
                rows_required=self.config.training_window,
            )
            return

        if (
            self.model is None
            or len(bars) - self.last_fit_bar_count
            >= self.config.retrain_interval
        ):
            self._fit_model(X_trainable, y_trainable, bars_seen=len(bars))
            self.logger.log(
                "model_refit",
                bar_time,
                bars_used=self.last_fit_bar_count,
            )

        feature_row = make_latest_feature_row(bars)
        if feature_row is None:
            return

        proba_up = float(self.model.predict_proba(feature_row.values)[0, 1])
        signal = 1 if proba_up >= self.config.probability_threshold else -1
        self.latest_probability = proba_up
        self.latest_signal = signal
        self.logger.log(
            "signal",
            bar_time,
            proba_up=round(proba_up, 6),
            signal=signal,
        )

        self._rebalance(session=session, timestamp=bar_time)

    def _completed_bars(self) -> pd.Series:
        """Resample ticks and drop the latest incomplete bar."""

        ticks = (
            pd.DataFrame(self.tick_history)
            .set_index("timestamp")
            .sort_index()
        )
        bars = (
            ticks["mid"]
            .resample(self.config.sample_interval)
            .last()
            .dropna()
        )
        if len(bars) <= 1:
            return pd.Series(dtype=float)
        return bars.iloc[:-1]

    def completed_bars(self) -> pd.Series:
        """Expose the latest completed bars for diagnostics and reporting."""

        return self._completed_bars().copy()

    def _fit_model(
        self,
        X_trainable: pd.DataFrame,
        y_trainable: pd.Series,
        bars_seen: int,
    ) -> None:
        """Fit the baseline logistic-regression model on the latest window."""

        X_train = X_trainable.iloc[-self.config.training_window:]
        y_train = y_trainable.iloc[-self.config.training_window:]
        self.model = build_logistic_regression_pipeline()
        self.model.fit(X_train.values, y_train.values)
        self.last_fit_bar_count = int(bars_seen)

    def _rebalance(
        self,
        session: TradingSession,
        timestamp: pd.Timestamp,
    ) -> None:
        """Move from the current position to the target signal position."""

        positions = session.broker.get_positions()
        current_quantity = (
            0.0 if not positions else float(positions[0]["quantity"])
        )
        target_quantity = self.config.position_size * float(self.latest_signal)
        delta = target_quantity - current_quantity
        if abs(delta) < 1e-12:
            self.logger.log(
                "hold",
                timestamp,
                target_quantity=target_quantity,
            )
            return

        side = "buy" if delta > 0.0 else "sell"
        receipt = session.place_market_order(
            symbol="EURUSD",
            side=side,
            quantity=abs(delta),
            meta={
                "signal": self.latest_signal,
                "proba_up": round(float(self.latest_probability), 6),
                "bar_timestamp": timestamp.isoformat(),
            },
        )
        snapshot = receipt["account_snapshot"]
        self.logger.log(
            "order_submitted",
            timestamp,
            side=side,
            quantity=abs(delta),
            fill_price=round(float(receipt["fill"]["price"]), 6),
            equity=round(float(snapshot["equity"]), 6),
        )


def build_monitoring_report(
    session: TradingSession,
    logger: DeploymentLogger,
) -> pd.Series:
    """Summarise the deployment run in a compact monitoring report."""

    logs = logger.to_frame()
    snapshot = session.broker.get_account_snapshot()
    signal_logs = logs.loc[logs["event"] == "signal", "signal"]

    return pd.Series(
        {
            "bars_processed": int((logs["event"] == "bar").sum()),
            "model_refits": int((logs["event"] == "model_refit").sum()),
            "orders_submitted": int((logs["event"] == "order_submitted").sum()),
            "last_signal": (
                int(signal_logs.iloc[-1]) if not signal_logs.empty else 0
            ),
            "final_equity": float(snapshot["equity"]),
            "realized_pnl": float(snapshot["realized_pnl"]),
            "open_positions": int(len(snapshot["positions"])),
        }
    )


def run_historical_engine_bridge() -> dict[str, object]:
    """Run the deployment architecture on historical EOD replay."""

    base_feed = HistoricalFeed.from_symbol(SYMBOL, spread_bps=1.0)
    prices = base_feed.prices.iloc[-260:]  # aligned EOD workflow
    feed = HistoricalFeed(
        symbol=base_feed.symbol,
        prices=prices,
        spread_bps=base_feed.spread_bps,
    )
    broker = PaperBroker(initial_cash=50_000.0, account_id="HIST-001")
    session = TradingSession(feed=feed, broker=broker)
    logger = DeploymentLogger(name="historical_bridge")
    deployer = BaselineMLDeployer(
        config=DeploymentConfig(
            sample_interval="1B",
            training_window=120,
            retrain_interval=20,
            position_size=10.0,
        ),
        logger=logger,
    )
    history = session.run(deployer.on_tick)
    report = build_monitoring_report(session=session, logger=logger)
    log_path = logger.write_jsonl(TMP_DIR / "historical_bridge.jsonl")
    logs = logger.to_frame()
    return {
        "history": history,
        "logs": logs,
        "bars": deployer.completed_bars(),
        "report": report,
        "log_path": log_path,
        "receipts": list(session.receipts),
    }


def run_simulated_live_deployment() -> dict[str, object]:
    """Run the same strategy on irregular simulated ticks."""

    feed = build_feed(
        symbol=SYMBOL,
        mode="simulated",
        periods=360,
        seed=11,
        start="2027-01-04 09:00:00",
        base_interval="1s",
        arrival_model="jittered",
        jitter_seconds=0.35,
        spread_bps=1.0,
    )
    broker = PaperBroker(initial_cash=50_000.0, account_id="SIM-001")
    session = TradingSession(feed=feed, broker=broker)
    logger = DeploymentLogger(name="simulated_live")
    deployer = BaselineMLDeployer(
        config=DeploymentConfig(
            sample_interval="5s",
            training_window=40,
            retrain_interval=10,
            position_size=5.0,
        ),
        logger=logger,
    )
    history = session.run(deployer.on_tick)
    report = build_monitoring_report(session=session, logger=logger)
    log_path = logger.write_jsonl(TMP_DIR / "simulated_live.jsonl")
    logs = logger.to_frame()
    return {
        "history": history,
        "logs": logs,
        "bars": deployer.completed_bars(),
        "report": report,
        "log_path": log_path,
        "receipts": list(session.receipts),
    }


def build_walkthrough_objects() -> dict[str, object]:
    """Create the deterministic objects used in the chapter walkthrough."""

    historical = run_historical_engine_bridge()
    simulated = run_simulated_live_deployment()
    return {
        "historical": historical,
        "simulated": simulated,
    }


def main() -> None:
    walkthrough = build_walkthrough_objects()
    historical = walkthrough["historical"]
    simulated = walkthrough["simulated"]

    assert historical["report"]["bars_processed"] > 0
    assert simulated["report"]["bars_processed"] > 0
    assert simulated["report"]["model_refits"] > 0

    print("SECTION: historical_report")
    print(historical["report"].round(6))

    print("\nSECTION: historical_bars_tail")
    print(historical["bars"].tail())

    print("\nSECTION: historical_model_cycle")
    mask = historical["logs"]["event"].isin(
        ["warmup", "model_refit", "signal", "order_submitted"]
    )
    cols = [
        "timestamp",
        "event",
        "bars_seen",
        "rows_available",
        "rows_required",
        "bars_used",
        "proba_up",
        "signal",
        "side",
        "quantity",
        "equity",
    ]
    print(historical["logs"].loc[mask].reindex(columns=cols).head(12))

    print("\nSECTION: historical_logs_tail")
    cols = ["timestamp", "event", "signal", "side", "quantity", "equity"]
    print(historical["logs"].reindex(columns=cols).tail(5))

    print("\nSECTION: historical_first_receipt")
    print(json.dumps(historical["receipts"][0], indent=2))

    print("\nSECTION: simulated_report")
    print(simulated["report"].round(6))

    print("\nSECTION: simulated_bars_head")
    print(simulated["bars"].head(10))

    print("\nSECTION: simulated_history_head")
    cols = ["bid", "ask", "mid", "cash", "equity", "position_quantity"]
    print(simulated["history"][cols].head())

    print("\nSECTION: simulated_model_cycle")
    mask = simulated["logs"]["event"].isin(
        ["warmup", "model_refit", "signal", "order_submitted"]
    )
    cols = [
        "timestamp",
        "event",
        "bars_seen",
        "rows_available",
        "rows_required",
        "bars_used",
        "proba_up",
        "signal",
        "side",
        "quantity",
        "equity",
    ]
    print(simulated["logs"].loc[mask].reindex(columns=cols).tail(12))

    print("\nSECTION: simulated_logs_tail")
    cols = [
        "timestamp",
        "event",
        "bars_seen",
        "proba_up",
        "signal",
        "side",
        "quantity",
        "equity",
    ]
    print(simulated["logs"].reindex(columns=cols).tail(8))

    print("\nSECTION: simulated_last_receipt")
    print(json.dumps(simulated["receipts"][-1], indent=2))

    print("\nSECTION: log_paths")
    print(historical["log_path"])
    print(simulated["log_path"])


if __name__ == "__main__":
    main()
