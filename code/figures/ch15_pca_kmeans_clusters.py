"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 15 - Machine and Deep Learning.

Unsupervised learning with PCA and K-means clustering on a synthetic dataset.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs


def main() -> None:
    """Visualize K-means clustering results before and after PCA."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    X, y_true = make_blobs(
        n_samples=600,
        n_features=3,
        centers=4,
        cluster_std=1.3,
        random_state=2027,
    )

    kmeans = KMeans(n_clusters=4, random_state=2027, n_init=10)
    labels = kmeans.fit_predict(X)

    pca = PCA(n_components=2, random_state=2027)
    X_pca = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

    ax = axes[0]
    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        cmap="tab10",
        s=15,
        alpha=0.8,
        edgecolor="k",
        linewidth=0.2,
    )
    ax.set_title("K-means Clusters (First Two Features)")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True, linestyle="--", alpha=0.3)

    ax = axes[1]
    ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        cmap="tab10",
        s=15,
        alpha=0.8,
        edgecolor="k",
        linewidth=0.2,
    )
    ax.set_title("K-means Clusters in First Two PC Scores")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle="--", alpha=0.3)

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch15_pca_kmeans_clusters.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
