"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 24 - Building a Market and Broker for Trading.

Market data feeds for the example trading engine package.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from engine.models import Tick
else:
    from .models import Tick


ROOT = Path(__file__).resolve().parents[2]
DATA_EOD = ROOT / "data" / "eod_data.csv"  # shared chapter dataset
TRADING_DAYS = 252  # standard annualisation convention
SECONDS_PER_YEAR = 365.25 * 24.0 * 60.0 * 60.0  # clock-time scaling
DEFAULT_SPREAD_BPS = 1.0  # narrow default spread for demo quotes
DEFAULT_BASE_INTERVAL = "1B"  # business-day spacing for chapter 24
DEFAULT_ARRIVAL_MODEL = "fixed"  # deterministic default arrival timing


def load_eod_data(path: Path = DATA_EOD) -> pd.DataFrame:
    """Load the example end-of-day dataset."""

    if not path.exists():
        raise FileNotFoundError(f"EOD dataset not found at {path}.")
    return pd.read_csv(
        path,
        parse_dates=["Date"],
        index_col="Date",
    ).sort_index()  # enforce monotonic time order


def load_symbol_history(symbol: str, path: Path = DATA_EOD) -> pd.Series:
    """Load one symbol from the example end-of-day dataset."""

    prices = load_eod_data(path)
    if symbol not in prices.columns:
        raise KeyError(f"symbol {symbol!r} not found in EOD dataset.")
    return prices[symbol].dropna()


def estimate_gbm_parameters(prices: pd.Series) -> tuple[float, float]:
    """Estimate annualised GBM drift and volatility from log returns."""

    log_returns = np.log(prices / prices.shift(1)).dropna()  # log-return path
    if log_returns.empty:
        raise ValueError("not enough data to estimate GBM parameters.")
    mu = float(log_returns.mean() * TRADING_DAYS)  # annualised drift
    sigma = float(
        log_returns.std(ddof=0) * np.sqrt(TRADING_DAYS)
    )  # annualised vol
    return mu, sigma


def make_tick(
    timestamp: pd.Timestamp,
    symbol: str,
    mid: float,
    spread_bps: float = DEFAULT_SPREAD_BPS,
    source: str = "historical",
) -> Tick:
    """Construct a bid-ask tick from a mid price and spread in basis points."""

    half_spread = mid * spread_bps / 10_000.0 / 2.0  # bps -> price spread
    return Tick(
        timestamp=pd.Timestamp(timestamp),
        symbol=symbol,
        bid=float(mid - half_spread),
        ask=float(mid + half_spread),
        source=source,
    )


def _to_timedelta(value: str | pd.Timedelta) -> pd.Timedelta:
    """Convert supported interval values to a pandas Timedelta."""

    if isinstance(value, pd.Timedelta):
        return value
    return pd.to_timedelta(value)


def generate_simulated_timestamps(
    start: pd.Timestamp,
    periods: int,
    base_interval: str | pd.Timedelta = DEFAULT_BASE_INTERVAL,
    arrival_model: str = DEFAULT_ARRIVAL_MODEL,
    jitter_seconds: float = 0.0,
    seed: int = 7,
) -> pd.DatetimeIndex:
    """Generate fixed or randomised timestamps for simulated ticks."""

    start_ts = pd.Timestamp(start)  # normalize timestamp input once
    if periods < 1:
        raise ValueError("periods must be at least 1.")

    if arrival_model == "fixed" and (
        isinstance(base_interval, str) and base_interval.upper().endswith("B")
    ):
        return pd.bdate_range(
            start_ts,
            periods=periods,
            freq=base_interval,
        )  # preserve business-day cadence

    base_delta = _to_timedelta(base_interval)
    if base_delta <= pd.Timedelta(0):
        raise ValueError("base_interval must be positive.")
    if jitter_seconds < 0.0:
        raise ValueError("jitter_seconds must be non-negative.")

    rng = np.random.default_rng(seed)  # deterministic if seed is fixed
    timestamps = [start_ts]  # first tick arrives at the chosen start

    for _ in range(1, periods):
        if arrival_model == "fixed":
            step_seconds = base_delta.total_seconds()  # exact spacing
        elif arrival_model == "jittered":
            jitter = float(
                rng.uniform(-jitter_seconds, jitter_seconds)
            )  # bounded timing noise
            step_seconds = max(
                0.001,
                base_delta.total_seconds() + jitter,
            )  # keep strictly increasing times
        elif arrival_model == "random":
            mean_seconds = base_delta.total_seconds()  # exponential mean gap
            step_seconds = max(
                0.001,
                float(rng.exponential(mean_seconds)),
            )  # Poisson-style arrivals
        else:
            raise ValueError(
                "arrival_model must be one of 'fixed', 'jittered', or 'random'."
            )

        timestamps.append(
            timestamps[-1] + pd.to_timedelta(step_seconds, unit="s")
        )

    return pd.DatetimeIndex(timestamps)


@dataclass(slots=True)
class HistoricalFeed:
    """Replay historical end-of-day prices as pseudo-real-time quote ticks."""

    symbol: str
    prices: pd.Series
    spread_bps: float = DEFAULT_SPREAD_BPS

    @classmethod
    def from_symbol(
        cls,
        symbol: str,
        spread_bps: float = DEFAULT_SPREAD_BPS,
    ) -> "HistoricalFeed":
        return cls(
            symbol=symbol,
            prices=load_symbol_history(symbol),
            spread_bps=spread_bps,
        )

    def __iter__(self) -> Iterator[Tick]:
        for timestamp, close in self.prices.items():
            yield make_tick(
                timestamp=pd.Timestamp(timestamp),
                symbol=self.symbol,
                mid=float(close),  # historical close used as synthetic mid
                spread_bps=self.spread_bps,  # constant demo spread
                source="historical",
            )


@dataclass(slots=True)
class GBMFeed:
    """Generate simulated quote ticks from estimated GBM dynamics."""

    symbol: str
    start: pd.Timestamp
    periods: int
    initial_price: float
    mu: float
    sigma: float
    seed: int = 7
    spread_bps: float = DEFAULT_SPREAD_BPS
    base_interval: str | pd.Timedelta = DEFAULT_BASE_INTERVAL
    arrival_model: str = DEFAULT_ARRIVAL_MODEL
    jitter_seconds: float = 0.0

    @classmethod
    def from_history(
        cls,
        symbol: str,
        periods: int = 252,
        seed: int = 7,
        spread_bps: float = DEFAULT_SPREAD_BPS,
        start: str | pd.Timestamp | None = None,
        base_interval: str | pd.Timedelta = DEFAULT_BASE_INTERVAL,
        arrival_model: str = DEFAULT_ARRIVAL_MODEL,
        jitter_seconds: float = 0.0,
    ) -> "GBMFeed":
        prices = load_symbol_history(symbol)  # calibration sample
        mu, sigma = estimate_gbm_parameters(prices)  # estimate GBM inputs
        if start is None:
            if (
                isinstance(base_interval, str)
                and base_interval.upper().endswith("B")
            ):
                start_ts = (
                    pd.Timestamp(prices.index[-1])
                    + pd.offsets.BusinessDay(1)
                )  # next business date
            else:
                start_ts = (
                    pd.Timestamp(prices.index[-1])
                    + _to_timedelta(base_interval)
                )  # next simulated timestamp
        else:
            start_ts = pd.Timestamp(start)  # explicit override from caller
        return cls(
            symbol=symbol,
            start=start_ts,
            periods=periods,
            initial_price=float(prices.iloc[-1]),
            mu=mu,
            sigma=sigma,
            seed=seed,
            spread_bps=spread_bps,
            base_interval=base_interval,
            arrival_model=arrival_model,
            jitter_seconds=jitter_seconds,
        )

    def __iter__(self) -> Iterator[Tick]:
        rng = np.random.default_rng(self.seed)  # one RNG for timing + prices
        timestamps = generate_simulated_timestamps(
            start=self.start,
            periods=self.periods,
            base_interval=self.base_interval,
            arrival_model=self.arrival_model,
            jitter_seconds=self.jitter_seconds,
            seed=self.seed,
        )

        level = self.initial_price  # evolve the simulated mid price
        previous_timestamp: pd.Timestamp | None = None
        for timestamp in timestamps:
            if previous_timestamp is None:
                dt = 1.0 / TRADING_DAYS  # sensible first-step fallback
            else:
                elapsed_seconds = (
                    pd.Timestamp(timestamp) - previous_timestamp
                ).total_seconds()  # actual simulated time gap
                dt = max(
                    elapsed_seconds / SECONDS_PER_YEAR,
                    1e-12,
                )  # keep diffusion stable
            shock = float(rng.standard_normal())  # one Brownian innovation
            level *= float(
                np.exp(
                    (self.mu - 0.5 * self.sigma ** 2) * dt
                    + self.sigma * np.sqrt(dt) * shock
                )
            )
            yield make_tick(
                timestamp=pd.Timestamp(timestamp),
                symbol=self.symbol,
                mid=level,  # simulated mid price at this event time
                spread_bps=self.spread_bps,  # convert mid to bid/ask quote
                source="simulated",
            )
            previous_timestamp = pd.Timestamp(timestamp)  # advance clock


def build_feed(
    symbol: str,
    mode: str = "historical",
    periods: int = 252,
    seed: int = 7,
    spread_bps: float = DEFAULT_SPREAD_BPS,
    start: str | pd.Timestamp | None = None,
    base_interval: str | pd.Timedelta = DEFAULT_BASE_INTERVAL,
    arrival_model: str = DEFAULT_ARRIVAL_MODEL,
    jitter_seconds: float = 0.0,
) -> HistoricalFeed | GBMFeed:
    """Factory for historical replay or GBM simulation."""

    if mode == "historical":
        return HistoricalFeed.from_symbol(symbol=symbol, spread_bps=spread_bps)
    if mode == "simulated":
        return GBMFeed.from_history(
            symbol=symbol,
            periods=periods,
            seed=seed,
            spread_bps=spread_bps,
            start=start,
            base_interval=base_interval,
            arrival_model=arrival_model,
            jitter_seconds=jitter_seconds,
        )
    raise ValueError("mode must be either 'historical' or 'simulated'.")
