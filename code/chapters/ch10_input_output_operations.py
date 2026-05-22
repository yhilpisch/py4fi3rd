"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 10 - Input/Output Operations.

This module collects small helpers for the file-handling patterns used
in Chapter 10:

- plain text and pickle round-trips,
- NumPy binary formats,
- and simple pandas I/O with CSV, JSON, SQLite, and HDF5.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd


def scratch_root() -> Path:
    """Return the disposable scratch folder used by the chapter."""

    root = Path(__file__).resolve().parents[2] / "_tmp"
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_text_report(root: Path) -> Path:
    """Write a small CSV-style text report and return its path."""

    report_path = root / "simple_report.csv"
    header = "symbol,price,quantity,value\n"
    rows = [
        "AAPL,180.25,10,1802.50\n",
        "SPY,520.10,5,2600.50\n",
    ]
    with report_path.open("w", encoding="utf-8") as f:
        f.write(header)
        f.writelines(rows)
    return report_path


def round_trip_pickle(root: Path) -> dict[str, dict[str, float]]:
    """Serialize and deserialize a small nested dictionary."""

    positions = {
        "AAPL": {"price": 180.25, "quantity": 10},
        "SPY": {"price": 520.10, "quantity": 5},
    }
    pickle_path = root / "positions.pkl"
    with pickle_path.open("wb") as f:
        pickle.dump(positions, f)
    with pickle_path.open("rb") as f:
        loaded = pickle.load(f)
    assert loaded == positions
    return loaded


def simulate_returns(seed: int = 42) -> npt.NDArray[np.floating]:
    """Generate a synthetic daily-return series."""

    rng = np.random.default_rng(seed=seed)
    return rng.normal(loc=0.0003, scale=0.01, size=252)


def save_load_returns(
    root: Path, daily_returns: npt.NDArray[np.floating]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Persist a single array to .npy and read it back."""

    returns_path = root / "daily_returns.npy"
    np.save(returns_path, daily_returns)
    loaded_returns = np.load(returns_path)
    assert np.allclose(daily_returns, loaded_returns)
    return daily_returns, loaded_returns


def bundle_arrays(
    root: Path, daily_returns: npt.NDArray[np.floating]
) -> dict[str, npt.NDArray[np.floating]]:
    """Write several related arrays to one .npz archive."""

    prices = 100 * (1 + daily_returns).cumprod()
    summary = np.array([prices.min(), prices.max(), prices[-1]], dtype=float)

    npz_path = root / "returns_and_prices.npz"
    np.savez(
        npz_path,
        returns=daily_returns,
        prices=prices,
        summary=summary,
    )

    with np.load(npz_path) as data:
        return {name: data[name] for name in data.files}


def load_prices(source: str | Path) -> pd.DataFrame:
    """Load a price table with a datetime index."""

    prices = pd.read_csv(
        source,
        parse_dates=["Date"],
        index_col="Date",
    )
    prices.index.name = "Date"
    return prices


def csv_round_trip(prices: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Write a DataFrame to CSV and read it back."""

    csv_path = root / "eod_prices.csv"
    prices.to_csv(csv_path, index=True)
    reloaded = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    reloaded.index.name = "Date"
    assert reloaded.equals(prices)
    return reloaded


def count_rows_in_chunks(csv_path: Path, chunksize: int = 250) -> int:
    """Count rows in a CSV file by iterating over chunks."""

    total_rows = 0
    for chunk in pd.read_csv(
        csv_path,
        chunksize=chunksize,
        parse_dates=["Date"],
    ):
        total_rows += len(chunk)
    return total_rows


def json_round_trip(prices: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Write and read a small window of a DataFrame as JSON."""

    latest = prices.iloc[-5:].copy()
    json_path = root / "latest_prices.json"
    latest.to_json(json_path, orient="table", date_format="iso")

    loaded = pd.read_json(json_path, orient="table")
    assert loaded.equals(latest)
    return loaded


def sqlite_round_trip(prices: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Store a DataFrame in SQLite and read back a filtered view."""

    db_path = root / "eod_prices.sqlite"
    with sqlite3.connect(db_path) as con:
        prices.to_sql("prices", con=con, if_exists="replace")
        rows = pd.read_sql(
            "SELECT Date, AAPL, SPY FROM prices "
            "WHERE Date >= '2020-01-01'",
            con=con,
            parse_dates=["Date"],
        )
    return rows


def hdf5_round_trip(prices: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Store a DataFrame in HDF5 and read it back."""

    hdf_path = root / "eod_prices.h5"
    prices.to_hdf(hdf_path, key="prices", mode="w")
    with pd.HDFStore(hdf_path, mode="r") as store:
        stored_prices = store["prices"]
    assert stored_prices.equals(prices)
    return stored_prices


def synthetic_prices() -> pd.DataFrame:
    """Return a small price table that does not depend on external data."""

    idx = pd.to_datetime(
        [
            "2026-01-02",
            "2026-01-05",
            "2026-01-06",
            "2026-01-07",
            "2026-01-08",
        ]
    )
    return pd.DataFrame(
        {
            "AAPL": [180.0, 182.0, 181.5, 183.2, 184.0],
            "SPY": [520.0, 521.5, 522.0, 523.1, 524.2],
        },
        index=idx,
    )


def main() -> None:
    """Run the chapter helpers on small in-memory data sets."""

    root = scratch_root()

    report_path = write_text_report(root)
    assert report_path.read_text(encoding="utf-8").startswith("symbol,price")

    round_trip_pickle(root)

    daily_returns = simulate_returns()
    save_load_returns(root, daily_returns)
    bundle_arrays(root, daily_returns)

    prices = synthetic_prices()
    prices.index.name = "Date"

    csv_round_trip(prices, root)
    csv_path = root / "eod_prices.csv"
    assert count_rows_in_chunks(csv_path) == len(prices)

    json_round_trip(prices, root)
    sqlite_round_trip(prices, root)
    hdf5_round_trip(prices, root)


if __name__ == "__main__":
    main()
