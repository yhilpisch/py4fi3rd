"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 14 - Statistics.

Normality diagnostics for log index levels simulated from GBM.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def main() -> None:
    """Generate histogram + QQ plot for simulated GBM log levels."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    rng = np.random.default_rng(seed=2027)
    s0 = 100.0
    r = 0.02
    sigma = 0.2
    T = 1.0
    n_paths = 250_000

    z = rng.standard_normal(n_paths)
    s_T = s0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    log_s_T = np.log(s_T)

    mu_hat = float(log_s_T.mean())
    sigma_hat = float(log_s_T.std(ddof=1))

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))

    # Histogram with fitted normal PDF.
    ax = axes[0]
    counts, bins, _ = ax.hist(
        log_s_T, bins=80, density=True, color="tab:blue", alpha=0.7
    )
    x = np.linspace(bins[0], bins[-1], 300)
    pdf = stats.norm.pdf(x, loc=mu_hat, scale=sigma_hat)
    ax.plot(x, pdf, color="tab:red", linewidth=1.5, label="Fitted normal PDF")
    ax.set_title("Histogram of Simulated Log Levels")
    ax.set_xlabel("log $S_T$")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    # QQ plot.
    ax = axes[1]
    stats.probplot(log_s_T, dist="norm", sparams=(mu_hat, sigma_hat), plot=ax)
    ax.set_title("QQ Plot vs Fitted Normal")
    ax.grid(True, linestyle="--", alpha=0.3)

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch14_normality_gbm.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

