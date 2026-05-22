"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 12 - Mathematical Tools.

Two-parameter function surface used in regression and optimization examples.

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


def fm(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Example function with multiple local minima."""
    return np.sin(x) + 0.05 * x**2 + np.sin(y) + 0.05 * y**2


def main() -> None:
    """Render a 3D surface of the two-parameter example function."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    x = np.linspace(-10.0, 10.0, 80)
    y = np.linspace(-10.0, 10.0, 80)
    X, Y = np.meshgrid(x, y)
    Z = fm(X, Y)

    fig = plt.figure(figsize=(7.5, 4.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        rstride=2,
        cstride=2,
        cmap="coolwarm",
        linewidth=0.4,
        antialiased=True,
    )
    ax.set_title("Two-Parameter Objective Function")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    fig.colorbar(surf, shrink=0.6, aspect=16)

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch12_function_surface.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

