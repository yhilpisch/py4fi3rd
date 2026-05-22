"""Python for Finance, 3rd ed., O'Reilly (2026).
Lab 03 - Building a Market Data Pipeline with EODHD.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from datetime import datetime
from io import StringIO
import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "eodhd_us_equities.csv"
CREDS_FILE = Path(__file__).resolve().with_name("creds.py")
DEFAULT_SYMBOLS = ("AAPL", "MSFT", "NVDA", "SPY")
DEFAULT_START = date(2024, 1, 2)
DEFAULT_STOP = date(2025, 3, 31)


def _load_creds_module(path: Path) -> ModuleType:
    """Load a local ``creds.py`` module from a concrete file path."""
    spec = importlib.util.spec_from_file_location("lab03_eod_creds", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load credentials module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_api_key(creds_file: Path = CREDS_FILE) -> str:
    """Read the local EODHD API key from ``code/labs/creds.py``."""
    if not creds_file.is_file():
        raise FileNotFoundError(f"Credentials file not found: {creds_file}")
    module = _load_creds_module(creds_file)
    api_key = getattr(module, "eod_key", None)
    if not api_key:
        raise AttributeError("The credentials module must define 'eod_key'.")
    return str(api_key)


@dataclass
class EODHDClient:
    """Compact client for the EODHD HTTP API."""

    api_key: str
    base_url: str = "https://eodhd.com/api"

    @classmethod
    def from_creds(cls, creds_file: Path = CREDS_FILE) -> "EODHDClient":
        """Build a client from the local credentials module."""
        return cls(api_key=load_api_key(creds_file))

    @staticmethod
    def _as_date_string(
        value: str | date | datetime | None,
    ) -> str | None:
        """Convert supported date inputs to ``YYYY-MM-DD`` strings."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d")
        if isinstance(value, date):
            return value.strftime("%Y-%m-%d")
        raise TypeError("Date bounds must be strings, date, or datetime.")

    def get_eod(
        self,
        symbol: str,
        exchange: str = "US",
        period: str = "d",
        start: str | date | datetime | None = None,
        stop: str | date | datetime | None = None,
        order: str = "a",
    ) -> pd.DataFrame:
        """Retrieve one EOD price history and return it as a DataFrame."""
        if period not in {"d", "w", "m"}:
            raise ValueError("period must be one of 'd', 'w', or 'm'")
        if order not in {"a", "d"}:
            raise ValueError("order must be 'a' or 'd'")

        full_symbol = f"{symbol}.{exchange}"
        url = f"{self.base_url}/eod/{full_symbol}"
        params: dict[str, object] = {
            "api_token": self.api_key,
            "period": period,
            "order": order,
            "fmt": "csv",
        }
        start_str = self._as_date_string(start)
        stop_str = self._as_date_string(stop)
        if start_str is not None:
            params["from"] = start_str
        if stop_str is not None:
            params["to"] = stop_str

        response = requests.get(url, params=params, timeout=30.0)
        response.raise_for_status()
        frame = pd.read_csv(
            StringIO(response.text),
            parse_dates=[0],
            index_col=0,
        )
        frame.index = pd.to_datetime(frame.index)
        frame.index.name = "date"
        frame.columns = [str(col).lower() for col in frame.columns]
        return frame.sort_index()


def build_market_panel(
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    start: date = DEFAULT_START,
    stop: date = DEFAULT_STOP,
    exchange: str = "US",
    client: EODHDClient | None = None,
) -> pd.DataFrame:
    """Download multiple symbols and stack them into one long-form panel."""
    if client is None:
        client = EODHDClient.from_creds()

    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        history = client.get_eod(
            symbol=symbol,
            exchange=exchange,
            start=start,
            stop=stop,
        )
        history = history.reset_index()
        history.insert(0, "symbol", symbol)
        cols = [
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
        ]
        frames.append(history[cols])

    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


def save_sample_dataset(
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
    start: date = DEFAULT_START,
    stop: date = DEFAULT_STOP,
    exchange: str = "US",
) -> pd.DataFrame:
    """Download and store the fixed lab sample."""
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    panel = build_market_panel(
        symbols=symbols,
        start=start,
        stop=stop,
        exchange=exchange,
    )
    panel.to_csv(DATA_FILE, index=False)
    return panel


def load_sample_dataset() -> pd.DataFrame:
    """Load the stored lab sample from disk."""
    return pd.read_csv(DATA_FILE, parse_dates=["date"])


def refresh_or_load_sample() -> pd.DataFrame:
    """Refresh the sample when possible and fall back to the cached CSV."""
    try:
        return save_sample_dataset()
    except (FileNotFoundError, requests.RequestException):
        if DATA_FILE.is_file():
            return load_sample_dataset()
        raise


def close_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long-form panel into an adjusted-close matrix."""
    return panel.pivot(index="date", columns="symbol",
                       values="adjusted_close").sort_index()


def returns_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily returns from the adjusted-close matrix."""
    prices = close_matrix(panel)
    return prices.pct_change().dropna()


def latest_snapshot(panel: pd.DataFrame) -> pd.DataFrame:
    """Build a one-row-per-symbol snapshot from the most recent date."""
    last_date = panel["date"].max()
    snap = panel.loc[panel["date"] == last_date].copy()
    cols = ["symbol", "date", "adjusted_close", "volume"]
    return snap[cols].sort_values("symbol").reset_index(drop=True)


def main() -> None:
    """Refresh the sample dataset and print compact inspection tables."""
    panel = refresh_or_load_sample()
    prices = close_matrix(panel)
    returns = returns_matrix(panel)

    print("LATEST SNAPSHOT")
    print(latest_snapshot(panel).to_string(index=False))
    print()

    print("ADJUSTED CLOSES")
    print(prices.head(3).round(2).to_string())
    print()

    print("RETURN CORRELATIONS")
    print(returns.corr().round(3).to_string())


if __name__ == "__main__":
    main()
