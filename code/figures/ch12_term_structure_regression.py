"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 12 - Mathematical Tools.

Synthetic term-structure function and regression approximations.

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


def term_structure(x: np.ndarray) -> np.ndarray:
    """Synthetic curve with linear and sinusoidal components."""
    return 0.5 * np.sin(x) + 0.1 * x


def main() -> None:
    """Plot the synthetic term structure and two regression fits."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    rng = np.random.default_rng(seed=42)
    x = np.linspace(0.0, 10.0, 50)
    y_clean = term_structure(x)
    noise = 0.05 * rng.standard_normal(x.shape[0])
    y_obs = y_clean + noise

    # Linear regression: intercept + slope.
    X_lin = np.column_stack([np.ones_like(x), x])
    beta_lin, *_ = np.linalg.lstsq(X_lin, y_obs, rcond=None)
    y_lin = X_lin @ beta_lin

    # Basis-function regression: intercept, x, x**2, sin(x).
    X_bf = np.column_stack(
        [
            np.ones_like(x),
            x,
            x**2,
            np.sin(x),
        ]
    )
    beta_bf, *_ = np.linalg.lstsq(X_bf, y_obs, rcond=None)
    y_bf = X_bf @ beta_bf

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(
        x, y_clean, color="black", linewidth=1.5, label="True term structure"
    )
    ax.scatter(
        x, y_obs, color="tab:gray", s=20, alpha=0.7, label="Noisy observations"
    )
    ax.plot(
        x,
        y_lin,
        color="tab:orange",
        linestyle="--",
        linewidth=1.25,
        label="Linear regression",
    )
    ax.plot(
        x,
        y_bf,
        color="tab:blue",
        linestyle="-.",
        linewidth=1.25,
        label="Basis-function regression",
    )

    ax.set_xlabel("Maturity")
    ax.set_ylabel("Value")
    ax.set_title("Term-Structure Approximation by Regression")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch12_term_structure_regression.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
