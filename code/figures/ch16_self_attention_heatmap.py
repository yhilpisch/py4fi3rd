"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 16 - NLP and LLM Foundations.

Self-attention heatmap for a toy finance-style headline.

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


def _compute_attention() -> tuple[np.ndarray, list[str]]:
    """Compute toy single-head self-attention weights."""
    tokens = ["rates", "cut", "to", "boost", "equities"]
    d_model = 4
    rng = np.random.default_rng(2027)

    x = rng.normal(scale=0.5, size=(len(tokens), d_model))

    w_q = rng.normal(scale=0.5, size=(d_model, d_model))
    w_k = rng.normal(scale=0.5, size=(d_model, d_model))

    q = x @ w_q
    k = x @ w_k

    scores = q @ k.T / np.sqrt(d_model)

    scores = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(scores)
    attn = e / e.sum(axis=-1, keepdims=True)
    return attn, tokens


def main() -> None:
    """Render a heatmap of toy self-attention weights."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7,
            "figure.dpi": 300,
        }
    )

    attn, tokens = _compute_attention()

    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    im = ax.imshow(attn, cmap="viridis", vmin=0.0, vmax=attn.max())

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=30, ha="right", fontsize=7)
    ax.set_yticklabels(tokens, fontsize=7)
    ax.set_xlabel("Key / value tokens", fontsize=7)
    ax.set_ylabel("Query tokens", fontsize=7)
    ax.set_title("Toy self-attention weights", fontsize=9)
    ax.tick_params(axis="both", labelsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention weight", fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch16_self_attention_heatmap.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
