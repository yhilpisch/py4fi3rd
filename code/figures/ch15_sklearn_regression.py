"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 15 - Machine and Deep Learning.

Baseline regression models on a synthetic dataset using scikit-learn.

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
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def main() -> None:
    """Compare linear and random-forest regression on noisy data."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    X, y, coef = make_regression(
        n_samples=400,
        n_features=1,
        n_informative=1,
        noise=10.0,
        coef=True,
        random_state=2027,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=2027,
    )

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_lin = lin_reg.predict(X_test)
    mse_lin = mean_squared_error(y_test, y_lin)

    rf_reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=2027,
    )
    rf_reg.fit(X_train, y_train)
    y_rf = rf_reg.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_rf)

    # Smooth x-grid for fitted curves.
    x_grid = np.linspace(X.min() - 1.0, X.max() + 1.0, 400).reshape(-1, 1)
    y_lin_grid = lin_reg.predict(x_grid)
    y_rf_grid = rf_reg.predict(x_grid)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.scatter(
        X_train[:, 0],
        y_train,
        color="tab:blue",
        alpha=0.6,
        label="Train",
        edgecolor="k",
        linewidth=0.3,
    )
    ax.scatter(
        X_test[:, 0],
        y_test,
        color="tab:orange",
        alpha=0.7,
        label="Test",
        edgecolor="k",
        linewidth=0.3,
    )
    ax.plot(
        x_grid[:, 0],
        y_lin_grid,
        color="tab:green",
        linewidth=2.0,
        label=f"Linear regression (MSE={mse_lin:.1f})",
    )
    ax.plot(
        x_grid[:, 0],
        y_rf_grid,
        color="tab:red",
        linewidth=2.0,
        linestyle="--",
        label=f"Random forest (MSE={mse_rf:.1f})",
    )

    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
    ax.set_title("Linear vs Random-Forest Regression on Synthetic Data")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch15_sklearn_regression.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
