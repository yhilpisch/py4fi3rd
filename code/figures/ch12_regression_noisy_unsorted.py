"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 12 - Mathematical Tools.

Regression with noisy and unsorted data.

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
    """Show how least-squares regression handles noisy and unsorted data."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    rng = np.random.default_rng(seed=123)
    x = np.linspace(0.0, 10.0, 50)
    y_true = term_structure(x)

    # Noisy observations on a regular grid.
    noise_y = 0.15 * rng.standard_normal(x.shape[0])
    y_noisy = y_true + noise_y

    X = np.column_stack([np.ones_like(x), x, x**2, np.sin(x)])
    beta, *_ = np.linalg.lstsq(X, y_noisy, rcond=None)
    y_fit = X @ beta

    # Shuffled x-values with the same underlying curve.
    perm = rng.permutation(x.shape[0])
    x_shuffled = x[perm]
    y_shuffled = y_true[perm] + 0.15 * rng.standard_normal(x.shape[0])

    X_shuffled = np.column_stack(
        [
            np.ones_like(x_shuffled),
            x_shuffled,
            x_shuffled**2,
            np.sin(x_shuffled),
        ]
    )
    beta_shuffled, *_ = np.linalg.lstsq(X_shuffled, y_shuffled, rcond=None)
    y_fit_shuffled = X_shuffled @ beta_shuffled

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharey=True)

    ax = axes[0]
    ax.plot(x, y_true, color="black", linewidth=1.5, label="True function")
    ax.scatter(
        x, y_noisy, color="tab:gray", s=20, alpha=0.7, label="Noisy data"
    )
    ax.plot(x, y_fit, color="tab:blue", linewidth=1.25, label="Regression fit")
    ax.set_title("Noisy Data (Sorted)")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(x, y_true, color="black", linewidth=1.5, label="True function")
    ax.scatter(
        x_shuffled,
        y_shuffled,
        color="tab:gray",
        s=20,
        alpha=0.7,
        label="Noisy, unsorted data",
    )
    order = np.argsort(x_shuffled)
    ax.plot(
        np.sort(x_shuffled),
        y_fit_shuffled[order],
        color="tab:orange",
        linewidth=1.25,
        label="Regression fit",
    )
    ax.set_title("Noisy Data (Unsorted)")
    ax.set_xlabel("x")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch12_regression_noisy_unsorted.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
