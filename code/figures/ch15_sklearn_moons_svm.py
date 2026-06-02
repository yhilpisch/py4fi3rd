"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 15 - Machine and Deep Learning.

Nonlinear SVM classification on synthetic two-moons data.

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
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def main() -> None:
    """Train an RBF SVM on two moons and plot the boundary."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    X, y = make_moons(n_samples=500, noise=0.25, random_state=2027)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=2027,
        stratify=y,
    )

    svm = SVC(kernel="rbf", gamma="scale", C=1.0, probability=True)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = svm.predict_proba(grid)[:, 1].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    contour = ax.contourf(
        xx,
        yy,
        zz,
        levels=np.linspace(0.0, 1.0, 21),
        cmap="coolwarm",
        alpha=0.7,
    )
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Class 1 probability (SVM)")

    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap="coolwarm",
        edgecolor="k",
        linewidth=0.4,
        alpha=0.9,
        label="Test points",
    )
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(f"RBF SVM on Two-Moons Dataset (acc={acc:.3f})")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch15_sklearn_moons_svm.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
