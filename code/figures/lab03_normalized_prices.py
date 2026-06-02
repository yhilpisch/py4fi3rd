"""Python for Finance, 3rd ed., O'Reilly (2026).
Lab 03 - Building a Market Data Pipeline with EODHD.

Figure: Normalized adjusted-close paths from the stored EODHD sample.

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
from code.labs.lab03_eodhd_pipeline import close_matrix
from code.labs.lab03_eodhd_pipeline import load_sample_dataset


SYMBOLS = DEFAULT_SYMBOLS


def main() -> None:
    """Generate the normalized adjusted-close comparison figure."""
    mpl.use("Agg", force=True)
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})

    panel = load_sample_dataset()
    prices = close_matrix(panel)[list(SYMBOLS)]
    normalized = prices / prices.iloc[0] * 100.0

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for symbol in SYMBOLS:
        ax.plot(normalized.index, normalized[symbol], label=symbol)
    ax.set_title("Normalized adjusted closes from the stored sample")
    ax.set_ylabel("Index level")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(ncol=2)

    figures_dir = ROOT / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "lab03_normalized_prices.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
