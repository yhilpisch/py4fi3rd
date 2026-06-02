"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 14 - Statistics.

Random portfolios and approximate mean-variance efficient frontier.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as sco


def main() -> None:
    """Generate random-portfolio cloud and efficient frontier."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    data_path = base_dir / "data" / "eod_data.csv"

    url = "https://hilpisch.com/eod_data.csv"

    try:
        data = pd.read_csv(
            data_path, index_col="Date", parse_dates=True
        ).dropna()
    except FileNotFoundError:
        data = pd.read_csv(url, index_col="Date", parse_dates=True).dropna()

    symbols = ["AAPL", "NVDA", "SPY", "GLD"]
    prices = data[symbols].dropna()

    rets = np.log(prices / prices.shift(1)).dropna()
    mean_rets = rets.mean() * 252
    cov_matrix = rets.cov() * 252

    noa = len(symbols)

    def port_ret(weights: np.ndarray) -> float:
        return float(np.sum(mean_rets * weights))

    def port_vol(weights: np.ndarray) -> float:
        return float(np.sqrt(weights.T @ cov_matrix @ weights))

    # Random portfolios.
    n_portfolios = 2_500
    rng = np.random.default_rng(seed=2027)
    prets = np.empty(n_portfolios)
    pvols = np.empty(n_portfolios)

    for i in range(n_portfolios):
        w = rng.random(noa)
        w /= np.sum(w)
        prets[i] = port_ret(w)
        pvols[i] = port_vol(w)

    # Efficient frontier for a grid of target returns.
    def efficient_weights(
        target: float,
        initial: np.ndarray | None = None,
    ) -> np.ndarray:
        if initial is None:
            initial = np.repeat(1.0 / noa, noa)
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w: port_ret(w) - target},
        )
        bounds = tuple((0.0, 1.0) for _ in range(noa))
        result = sco.minimize(
            port_vol,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if not result.success:
            msg = f"frontier optimization failed for target={target:.6f}"
            raise RuntimeError(msg)
        return np.asarray(result.x, dtype=float)

    # Start at global minimum variance to avoid the inefficient lower branch.
    gmv = sco.minimize(
        port_vol,
        np.repeat(1.0 / noa, noa),
        method="SLSQP",
        bounds=tuple((0.0, 1.0) for _ in range(noa)),
        constraints=({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},),
    )
    if not gmv.success:
        raise RuntimeError("global minimum-variance optimization failed")

    gmv_weights = np.asarray(gmv.x, dtype=float)
    gmv_return = port_ret(gmv_weights)
    target_returns = np.linspace(gmv_return, prets.max(), 50)

    frontier_weights = []
    initial = gmv_weights
    for target in target_returns:
        initial = efficient_weights(float(target), initial=initial)
        frontier_weights.append(initial)
    frontier_weights = np.asarray(frontier_weights)
    frontier_vols = np.array(
        [port_vol(w) for w in frontier_weights], dtype=float
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    scatter = ax.scatter(
        pvols,
        prets,
        c=prets / pvols,
        marker=".",
        alpha=0.8,
        cmap="coolwarm",
        label="Random portfolios",
    )
    ax.plot(
        frontier_vols,
        target_returns,
        "b",
        linewidth=2.5,
        label="Efficient frontier",
    )
    ax.set_xlabel("expected volatility")
    ax.set_ylabel("expected return")
    ax.set_title("Random Portfolios and Efficient Frontier")
    ax.grid(True, linestyle="--", alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Sharpe ratio")
    ax.legend(loc="best")

    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch14_random_portfolios_frontier.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
