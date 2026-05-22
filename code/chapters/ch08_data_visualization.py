"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 8 - Data Visualization.

This module complements the figure-generation scripts by showing
how you might structure plotting code in a slightly more modular,
production-oriented style. It focuses on:

- typed helpers that accept precomputed data,
- separation between data creation and plotting,
- and small invariants around input shapes.

For full-sized figures used in the book, see the scripts under
``code/figures/``.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def configure_style() -> None:
    """Apply the baseline style used throughout the chapter."""

    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})


def line_plot(ax: plt.Axes, data: npt.NDArray[np.floating]) -> None:
    """Draw a simple line plot of a 1D array on the given ``Axes``."""

    series = np.asarray(data, dtype=float)
    ax.plot(series, color="tab:blue", linewidth=1.25)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True, linestyle="--", alpha=0.3)


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save a figure and create parent directories if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def demo_line_plot(outfile: Path) -> None:
    """Generate a small synthetic line plot and save it."""

    configure_style()
    rng = np.random.default_rng(seed=42)
    steps = rng.normal(loc=0.0, scale=0.01, size=100)
    prices = 100 * (1 + steps).cumprod()

    fig, ax = plt.subplots(figsize=(7.5, 4))
    line_plot(ax, prices)
    ax.set_title("Synthetic Price Series (Demo)")
    save_figure(fig, outfile)


def main() -> None:
    """Run the demo when executed as a script."""

    base_dir = Path(__file__).resolve().parents[2]
    out = base_dir / "assets" / "figures" / "ch08_demo_line.png"
    demo_line_plot(out)


if __name__ == "__main__":
    main()
