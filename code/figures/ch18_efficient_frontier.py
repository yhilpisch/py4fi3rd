"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 18 - Portfolio Construction and Risk.

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


def load_returns() -> pd.DataFrame:
    """Load daily returns for a small universe and benchmark."""

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    local = base_dir / "data" / "eod_data.csv"
    remote = "https://hilpisch.com/eod_data.csv"
    source: str | pathlib.Path = local if local.exists() else remote

    prices = pd.read_csv(source, parse_dates=["Date"], index_col="Date")

    universe = ["AAPL", "JPM", "TLT"]
    cols = universe + ["SPY"]
    sub = prices[cols].dropna(how="any")
    rets_all = sub.pct_change().dropna()

    max_rows = 2 * 252
    if len(rets_all) > max_rows:
        rets = rets_all.iloc[-max_rows:]
    else:
        rets = rets_all

    return rets[universe], rets["SPY"]


def main() -> None:
    """Plot an efficient frontier with reference portfolios."""

    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

    rets, r_bench = load_returns()

    mu_daily = rets.mean()
    cov_daily = rets.cov()

    mu_annual = (1.0 + mu_daily) ** 252 - 1.0
    cov_annual = cov_daily * 252.0

    Sigma = cov_annual.values
    ones = np.ones(len(mu_annual))
    inv_Sigma_ones = np.linalg.solve(Sigma, ones)
    w_gmv = inv_Sigma_ones / (ones @ inv_Sigma_ones)

    # Equal-weight portfolio
    w_eq = np.repeat(1.0 / len(mu_annual), len(mu_annual))

    def port_stats(weights: np.ndarray) -> tuple[float, float]:
        mu_p = float(weights @ mu_annual.values)
        var_p = float(weights @ (Sigma @ weights))
        return mu_p, np.sqrt(var_p)

    mu_eq, vol_eq = port_stats(w_eq)
    mu_gmv, vol_gmv = port_stats(w_gmv)

    # Random portfolios for a coarse frontier
    rng = np.random.default_rng(seed=2027)
    n_ports = 5000
    rand_w = rng.dirichlet(alpha=np.ones(len(mu_annual)), size=n_ports)

    mu_rand = rand_w @ mu_annual.values
    vol_rand = np.sqrt(np.einsum("ij,jk,ik->i", rand_w, Sigma, rand_w))

    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    ax.scatter(vol_rand, mu_rand, s=6, alpha=0.3, label="Random portfolios")
    ax.scatter(
        vol_eq,
        mu_eq,
        color="tab:blue",
        s=60,
        marker="o",
        label="Equal-weight",
    )
    ax.scatter(vol_gmv, mu_gmv, color="tab:red", s=60, marker="D", label="GMV")

    ax.set_xlabel("Annualised volatility")
    ax.set_ylabel("Annualised expected return")
    ax.set_title("Sample efficient frontier: equal-weight vs GMV")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch18_efficient_frontier.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
