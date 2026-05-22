"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 15 - Machine and Deep Learning.

Two-layer PyTorch MLP classifier on a synthetic two-moons dataset.

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
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from torch import nn

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None


class MLP(nn.Module):
    """Simple two-layer MLP for binary classification."""

    def __init__(self, in_features: int = 2, hidden_features: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 1),
        )

    def forward(  # type: ignore[override]
        self, x: torch.Tensor
    ) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    """Train a small PyTorch MLP and visualise its boundary."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    torch.manual_seed(2027)
    X, y = make_moons(n_samples=600, noise=0.25, random_state=2027)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=2027,
        stratify=y,
    )

    device = torch.device("cpu")

    model = MLP(in_features=2, hidden_features=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device).view(-1, 1)

    model.train()
    n_epochs = 500
    epochs = range(n_epochs)
    iterator = (
        tqdm(epochs, desc="Training MLP (two moons)", leave=False)
        if tqdm
        else epochs
    )
    for epoch in iterator:
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = loss_fn(logits, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).to(device)
        logits_test = model(X_test_t)
        probs_test = torch.sigmoid(logits_test).cpu().numpy().ravel()
        y_pred = (probs_test >= 0.5).astype(int)
        acc = (y_pred == y_test).mean()

    # Decision boundary grid.
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    with torch.no_grad():
        logits_grid = model(torch.from_numpy(grid).to(device))
        probs_grid = torch.sigmoid(logits_grid).cpu().numpy().ravel()
    zz = probs_grid.reshape(xx.shape)

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
    cbar.set_label("Class 1 probability (MLP)")

    scatter = ax.scatter(
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
    ax.set_title(f"PyTorch MLP on Two-Moons Dataset (acc={acc:.3f})")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch15_pytorch_moons_mlp.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
