"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 12 - Mathematical Tools.

Shaded area under the standard normal density.

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


def normal_pdf(x: np.ndarray) -> np.ndarray:
    """Standard normal probability density function."""
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x**2)


def main() -> None:
    """Visualize an integral as a shaded area under the normal PDF."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    x = np.linspace(-4.0, 4.0, 500)
    y = normal_pdf(x)

    a, b = -1.0, 1.0
    mask = (x >= a) & (x <= b)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(x, y, color="tab:blue", linewidth=1.5, label="Standard normal PDF")
    ax.fill_between(
        x[mask],
        0.0,
        y[mask],
        color="tab:blue",
        alpha=0.25,
        label="Area between a and b",
    )
    ax.axvline(a, color="tab:gray", linestyle="--", linewidth=1.0)
    ax.axvline(b, color="tab:gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_title("Integral as Shaded Area Under the Normal Density")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch12_normal_integral_area.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
