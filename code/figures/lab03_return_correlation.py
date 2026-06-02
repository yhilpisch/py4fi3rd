"""Python for Finance, 3rd ed., O'Reilly (2026).
Lab 03 - Building a Market Data Pipeline with EODHD.

Figure: Return correlations across the stored EODHD sample.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import os
import pathlib
import sys

os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"
os.environ["MPLBACKEND"] = "Agg"

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.labs.lab03_eodhd_pipeline import DEFAULT_SYMBOLS
from code.labs.lab03_eodhd_pipeline import load_sample_dataset
from code.labs.lab03_eodhd_pipeline import returns_matrix


SYMBOLS = DEFAULT_SYMBOLS


def main() -> None:
    """Generate a compact lower-triangle correlation figure."""
    mpl.use("Agg", force=True)
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7,
            "figure.dpi": 300,
        }
    )

    panel = load_sample_dataset()
    corr = returns_matrix(panel)[list(SYMBOLS)].corr()
    fig = plt.figure(figsize=(3.1, 2.6))
    grid = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[1.0, 0.055],
        left=0.18,
        right=0.86,
        bottom=0.16,
        top=0.92,
        wspace=0.18,
    )
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])
    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=0.2, vmax=1.0)

    n = len(SYMBOLS)
    for row in range(n):
        for col in range(n):
            if col > row:
                continue
            value = float(corr.iloc[row, col])
            color = cmap(norm(value))
            radius = 0.40 if row == col else 0.34
            circle = plt.Circle((col, row), radius, color=color, ec="none")
            ax.add_patch(circle)
            text_color = "black" if value > 0.82 else "white"
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=5.5,
            )

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(n))
    ax.set_xticklabels(SYMBOLS, fontsize=6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(SYMBOLS, fontsize=6)
    ax.tick_params(length=0, pad=2)

    for row in range(n + 1):
        ax.axhline(row - 0.5, color="0.88", lw=0.6, zorder=0)
        ax.axvline(row - 0.5, color="0.88", lw=0.6, zorder=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        cax=cax,
        ticks=[0.2, 0.5, 0.8, 1.0],
    )
    cbar.ax.tick_params(labelsize=5.5, length=0)
    cbar.outline.set_linewidth(0.4)

    figures_dir = ROOT / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "lab03_return_correlation.png"
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
