"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 15 - Machine and Deep Learning.

This companion module wraps the chapter's supervised, unsupervised, and simple
neural-network examples in reusable helpers with optional dependency checks.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

try:
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs, make_moons, make_regression
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import accuracy_score, mean_squared_error
except ImportError:  # pragma: no cover - optional dependency
    KMeans = None
    PCA = None
    RandomForestRegressor = None
    LinearRegression = None
    LogisticRegression = None
    accuracy_score = None
    mean_squared_error = None
    make_blobs = None
    make_moons = None
    make_regression = None

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None


Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class RegressionScores:
    linear_mse: float
    forest_mse: float


@dataclass(frozen=True)
class ClassificationScore:
    accuracy: float


def require_sklearn() -> None:
    """Raise a clear error if scikit-learn is unavailable."""

    if make_moons is None:
        raise RuntimeError('scikit-learn is required for chapter 15 helpers')


def moons_classification(seed: int = 2027) -> ClassificationScore:
    """Fit a logistic-regression baseline on a noisy two-moons dataset."""

    require_sklearn()
    X, y = make_moons(n_samples=500, noise=0.25, random_state=seed)
    rng = np.random.default_rng(seed=seed)
    idx = rng.permutation(X.shape[0])
    n_train = int(0.7 * X.shape[0])
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    clf = LogisticRegression(solver='lbfgs')
    clf.fit(X[train_idx], y[train_idx])
    pred = clf.predict(X[test_idx])
    return ClassificationScore(
        accuracy=float(accuracy_score(y[test_idx], pred))
    )


def regression_baselines(seed: int = 2027) -> RegressionScores:
    """Compare linear and random-forest regression on synthetic data."""

    require_sklearn()
    X_reg, y_reg = make_regression(
        n_samples=400,
        n_features=1,
        n_informative=1,
        noise=10.0,
        random_state=seed,
    )
    rng = np.random.default_rng(seed=seed)
    idx = rng.permutation(X_reg.shape[0])
    n_train = int(0.7 * X_reg.shape[0])
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    lin = LinearRegression().fit(X_reg[train_idx], y_reg[train_idx])
    forest = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=seed,
    )
    forest.fit(X_reg[train_idx], y_reg[train_idx])

    linear_mse = float(
        mean_squared_error(y_reg[test_idx], lin.predict(X_reg[test_idx]))
    )
    forest_mse = float(
        mean_squared_error(y_reg[test_idx], forest.predict(X_reg[test_idx]))
    )
    return RegressionScores(linear_mse=linear_mse, forest_mse=forest_mse)


def pca_kmeans(seed: int = 2027) -> tuple[Array, Array, Array]:
    """Project synthetic blobs with PCA and cluster them with K-means."""

    require_sklearn()
    X_blob, _ = make_blobs(
        n_samples=600,
        n_features=3,
        centers=4,
        cluster_std=1.3,
        random_state=seed,
    )
    labels = KMeans(
        n_clusters=4,
        random_state=seed,
        n_init=10,
    ).fit_predict(X_blob)
    X_pca = PCA(n_components=2, random_state=seed).fit_transform(X_blob)
    return X_blob.astype(float), labels.astype(float), X_pca.astype(float)


def require_torch() -> None:
    """Raise a clear error if PyTorch is unavailable."""

    if torch is None or nn is None:
        raise RuntimeError(
            'PyTorch is required for the neural-network chapter helpers'
        )


def torch_moons_mlp(seed: int = 2027, epochs: int = 200) -> float:
    """Train a small MLP on the two-moons dataset and return test accuracy."""

    require_sklearn()
    require_torch()

    X, y = make_moons(n_samples=500, noise=0.25, random_state=seed)
    rng = np.random.default_rng(seed=seed)
    idx = rng.permutation(X.shape[0])
    n_train = int(0.7 * X.shape[0])
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    torch.manual_seed(seed)
    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X[test_idx], dtype=torch.float32)
    y_test = torch.tensor(y[test_idx], dtype=torch.float32).reshape(-1, 1)

    model = nn.Sequential(
        nn.Linear(2, 16),
        nn.Tanh(),
        nn.Linear(16, 16),
        nn.Tanh(),
        nn.Linear(16, 1),
        nn.Sigmoid(),
    )
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred_test = (model(X_test) >= 0.5).float()
        accuracy = (pred_test.eq(y_test).float().mean()).item()
    return float(accuracy)


def main() -> None:
    """Run a compact set of chapter-style ML demos."""

    if make_moons is not None:
        cls_score = moons_classification()
        assert 0.0 <= cls_score.accuracy <= 1.0
        print(cls_score)
        reg_scores = regression_baselines()
        assert reg_scores.linear_mse > 0.0
        assert reg_scores.forest_mse > 0.0
        print(reg_scores)
        _, labels, X_pca = pca_kmeans()
        assert labels.shape[0] == X_pca.shape[0]
        print(labels[:10])
        print(X_pca[:3])
    if torch is not None and make_moons is not None:
        nn_accuracy = torch_moons_mlp()
        assert 0.0 <= nn_accuracy <= 1.0
        print(nn_accuracy)


if __name__ == '__main__':
    main()
