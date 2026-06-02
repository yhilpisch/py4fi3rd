"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 15 - Machine and Deep Learning.

PyTorch LSTM classifier predicting short-horizon up/down moves
from lagged index returns.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None

SYMBOL = "BTC-USD"
WINDOW = 20
HORIZON = 5
HIDDEN_SIZE = 16
N_EPOCHS = 2500


class LSTMClassifier(nn.Module):
    """Small LSTM-based classifier for binary up/down prediction."""

    def __init__(self, input_size: int = 1, hidden_size: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(  # type: ignore[override]
        self, x: torch.Tensor
    ) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)


def _load_eod_data() -> pd.DataFrame:
    """Load end-of-day data with local + URL fallback."""
    base_dir = pathlib.Path(__file__).resolve().parents[2]
    local_path = base_dir / "data" / "eod_data.csv"
    url = "https://hilpisch.com/eod_data.csv"

    try:
        data = pd.read_csv(local_path, index_col="Date", parse_dates=True)
    except FileNotFoundError:
        data = pd.read_csv(url, index_col="Date", parse_dates=True)
    return data.dropna()


def _build_lagged_dataset(
    returns: pd.Series, window: int, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create lagged-return windows and up/down labels."""
    rets = returns.dropna().to_numpy(dtype=np.float32)
    dates = returns.dropna().index
    n_obs = rets.shape[0]
    if n_obs <= window + horizon:
        raise ValueError(
            "Not enough observations for the chosen window length."
        )

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    date_list: list[pd.Timestamp] = []
    for t in range(window, n_obs - horizon + 1):
        window_vals = rets[t - window : t]
        target_ret = rets[t : t + horizon].sum()
        X_list.append(window_vals.reshape(-1, 1))
        y_list.append(int(target_ret > 0.0))
        date_list.append(dates[t + horizon - 1])

    X = np.stack(X_list, axis=0)  # shape: (n_samples, window, 1)
    y = np.asarray(y_list, dtype=np.float32)  # shape: (n_samples,)
    return X, y, np.asarray(date_list)


def main() -> None:
    """Train an LSTM classifier on lagged returns."""
    mpl.style.use("seaborn-v0_8")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 300,
        }
    )

    torch.manual_seed(2027)

    data = _load_eod_data()
    symbol = SYMBOL
    prices = data[symbol].dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    X, y, dates = _build_lagged_dataset(
        log_returns, window=WINDOW, horizon=HORIZON
    )

    split = train_test_split(
        X,
        y,
        dates,
        test_size=0.3,
        shuffle=False,
    )
    X_train, X_test, y_train, y_test, dates_train, dates_test = split

    # Standardise lagged-return windows based on the training data only.
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

    device = torch.device("cpu")
    model = LSTMClassifier(input_size=1, hidden_size=HIDDEN_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Balance the up/down classes so the model does not collapse
    # to always predicting the majority class.
    pos = float(y_train.mean())
    pos_weight = (1.0 - pos) / pos
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32)
    )

    X_train_t = torch.from_numpy(X_train_scaled).to(device)
    y_train_t = torch.from_numpy(y_train).to(device).view(-1, 1)

    model.train()
    epochs = range(N_EPOCHS)
    iterator = (
        tqdm(epochs, desc="Training LSTM (classification)", leave=False)
        if tqdm
        else epochs
    )
    for epoch in iterator:
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = loss_fn(logits, y_train_t)
        loss.backward()
        optimizer.step()
        if tqdm is not None:
            iterator.set_postfix(loss=float(loss.item()))
    train_loss = float(loss.item())

    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test_scaled).to(device)
        logits_test = model(X_test_t)
        probs_test = torch.sigmoid(logits_test).cpu().numpy().ravel()
        y_pred = (probs_test >= 0.5).astype(int)
        acc = float((y_pred == y_test).mean())

    print(
        f"[LSTM classification] window={WINDOW}, horizon={HORIZON}, "
        f"train_loss={train_loss:.4f}, accuracy={acc:.3f}"
    )

    # Build time-series figure showing returns and the train/test split only.
    fig, ax = plt.subplots(figsize=(9.5, 3.0))
    ax.plot(
        log_returns.index,
        log_returns.values * 100.0,
        color="tab:blue",
        linewidth=0.8,
        label="Daily log return",
    )
    ax.axvline(
        dates_test[0],
        color="k",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
    )
    ax.set_ylabel("Daily log return (%)")
    ax.set_title(f"{symbol} daily returns with LSTM train/test split")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.3)

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    outfile_ts = figures_dir / "ch15_pytorch_timeseries_classification.png"
    fig.tight_layout()
    fig.savefig(outfile_ts, dpi=300)
    plt.close(fig)

    # Confusion matrix figure.
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig_cm, ax_cm = plt.subplots(figsize=(3.4, 2.6))
    im = ax_cm.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)

    for i in range(2):
        for j in range(2):
            ax_cm.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                ha="center",
                va="center",
                color="black",
                fontsize=7,
            )

    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Down (0)", "Up (1)"], rotation=20, fontsize=7)
    ax_cm.set_yticklabels(["Down (0)", "Up (1)"], fontsize=7)
    ax_cm.set_xlabel("Predicted label", fontsize=7)
    ax_cm.set_ylabel("True label", fontsize=7)
    ax_cm.tick_params(axis="both", labelsize=7)
    ax_cm.set_title(
        f"LSTM {HORIZON}-day up/down (acc={acc:.3f})",
        fontsize=8,
    )

    cbar = fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalised freq.", fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    outfile_cm = (
        figures_dir / "ch15_pytorch_timeseries_classification_confusion.png"
    )
    fig_cm.tight_layout(pad=0.8)
    fig_cm.savefig(outfile_cm, dpi=300)
    plt.close(fig_cm)


if __name__ == "__main__":
    main()
