"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 12 - Mathematical Tools.

Linear and cubic spline interpolation for a synthetic term structure.

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
from scipy.interpolate import CubicSpline, interp1d


def term_structure(x: np.ndarray) -> np.ndarray:
    """Synthetic curve with linear and sinusoidal components."""
    return 0.5 * np.sin(x) + 0.1 * x


def main() -> None:
    """Compare linear and cubic spline interpolation on a coarse grid."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    x_knots = np.linspace(0.0, 10.0, 12)
    y_knots = term_structure(x_knots)
    x_fine = np.linspace(0.0, 10.0, 400)

    # Linear interpolation via piecewise linear segments between knots.
    linear_interp = interp1d(x_knots, y_knots, kind="linear")
    y_lin = linear_interp(x_fine)

    # Cubic spline interpolation with default settings.
    spline_cubic = CubicSpline(x_knots, y_knots)
    y_cubic = spline_cubic(x_fine)

    y_true = term_structure(x_fine)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharey=True)

    ax = axes[0]
    ax.plot(x_fine, y_true, color="black", linewidth=1.5, label="True function")
    ax.plot(
        x_fine,
        y_lin,
        color="tab:orange",
        linewidth=1.25,
        label="Linear spline",
    )
    ax.scatter(
        x_knots, y_knots, color="tab:blue", s=25, zorder=3, label="Knots"
    )
    ax.set_title("Linear Spline Interpolation")
    ax.set_xlabel("Maturity")
    ax.set_ylabel("Value")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(x_fine, y_true, color="black", linewidth=1.5, label="True function")
    ax.plot(
        x_fine,
        y_cubic,
        color="tab:green",
        linewidth=1.25,
        label="Cubic spline",
    )
    ax.scatter(
        x_knots, y_knots, color="tab:blue", s=25, zorder=3, label="Knots"
    )
    ax.set_title("Cubic Spline Interpolation")
    ax.set_xlabel("Maturity")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch12_spline_interpolation.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
