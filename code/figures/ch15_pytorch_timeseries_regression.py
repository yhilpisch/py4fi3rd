"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 15 - Machine and Deep Learning.

PyTorch LSTM regressor predicting next-day returns
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None

SYMBOL = "SPY"
WINDOW = 10
HIDDEN_SIZE = 16
N_EPOCHS = 2500


class LSTMRegressor(nn.Module):
    """Small LSTM-based regressor for next-day returns."""

    def __init__(self, input_size: int = 1, hidden_size: int = 16) -> None:
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
    returns: pd.Series, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create lagged-return windows and next-day return targets."""
    rets = returns.dropna().to_numpy(dtype=np.float32)
    n_obs = rets.shape[0]
    if n_obs <= window:
        raise ValueError(
            "Not enough observations for the chosen window length."
        )

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    for t in range(window, n_obs):
        window_vals = rets[t - window : t]
        target_ret = rets[t]
        X_list.append(window_vals.reshape(-1, 1))
        y_list.append(float(target_ret))

    X = np.stack(X_list, axis=0)  # shape: (n_samples, window, 1)
    y = np.asarray(y_list, dtype=np.float32)  # shape: (n_samples,)
    dates = returns.dropna().index[window:]
    return X, y, dates.to_numpy()


def main() -> None:
    """Train an LSTM regressor on lagged returns."""
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

    X, y, dates = _build_lagged_dataset(log_returns, window=WINDOW)

    split = train_test_split(
        X,
        y,
        dates,
        test_size=0.3,
        shuffle=False,
    )
    X_train, X_test, y_train, y_test, dates_train, dates_test = split

    # Standardize lagged-return windows based on the training data only.
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

    device = torch.device("cpu")
    model = LSTMRegressor(input_size=1, hidden_size=HIDDEN_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    X_train_t = torch.from_numpy(X_train_scaled).to(device)
    y_train_t = torch.from_numpy(y_train).to(device).view(-1, 1)

    model.train()
    epochs = range(N_EPOCHS)
    iterator = (
        tqdm(epochs, desc="Training LSTM (regression)", leave=False)
        if tqdm
        else epochs
    )
    for epoch in iterator:
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = loss_fn(preds, y_train_t)
        loss.backward()
        optimizer.step()
        if tqdm is not None:
            iterator.set_postfix(loss=float(loss.item()))
    train_loss = float(loss.item())

    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test_scaled).to(device)
        preds_test = model(X_test_t).cpu().numpy().ravel()

    # Simple diagnostics.
    mse = float(((preds_test - y_test) ** 2).mean())
    corr = float(np.corrcoef(preds_test, y_test)[0, 1])

    print(
        f"[LSTM regression] window={WINDOW}, train_loss={train_loss:.6f}, "
        f"test_mse={mse:.6f}, test_corr={corr:.3f}"
    )

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 5.5))

    ax = axes[0]
    ax.plot(
        dates_test,
        y_test * 100.0,
        label="True",
        color="tab:blue",
        linewidth=0.8,
    )
    ax.plot(
        dates_test,
        preds_test * 100.0,
        label="Predicted",
        color="tab:red",
        linewidth=1.0,
        alpha=0.8,
    )
    ax.set_ylabel("Next-day return (%)")
    ax.set_title(
        f"{symbol} next-day return regression (MSE={mse:.5f}, corr={corr:.2f})"
    )
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.3)

    ax = axes[1]
    true_pct = y_test * 100.0
    pred_pct = preds_test * 100.0
    ax.scatter(
        true_pct,
        pred_pct,
        color="tab:purple",
        alpha=0.7,
        edgecolor="k",
        linewidth=0.3,
    )
    # Focus on the central region where most points lie to avoid extreme
    # outliers distorting the scale.
    max_abs = float(
        np.percentile(np.abs(np.concatenate([true_pct, pred_pct])), 99.0)
    )
    max_abs = max(max_abs, 1.0)
    ax.set_xlim(-max_abs, max_abs)
    ax.set_ylim(-max_abs, max_abs)
    lims = np.array([-max_abs, max_abs])
    ax.plot(lims, lims, "k--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("True next-day return (%)")
    ax.set_ylabel("Predicted next-day return (%)")
    ax.grid(True, linestyle="--", alpha=0.3)

    base_dir = pathlib.Path(__file__).resolve().parents[2]
    figures_dir = base_dir / "assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outfile = figures_dir / "ch15_pytorch_timeseries_regression.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)

    # Equity-curve comparison based on true vs predicted next-day log returns.
    eq_true = np.exp(np.cumsum(y_test))
    eq_pred = np.exp(np.cumsum(preds_test))
    eq_true /= eq_true[0]
    eq_pred /= eq_pred[0]

    fig_eq, ax_eq = plt.subplots(1, 1, figsize=(9.5, 3.0))
    ax_eq.plot(
        dates_test,
        eq_true,
        label="True equity curve",
        color="tab:blue",
        linewidth=1.0,
    )
    ax_eq.plot(
        dates_test,
        eq_pred,
        label="Predicted equity curve",
        color="tab:red",
        linewidth=1.0,
        alpha=0.9,
    )
    ax_eq.set_ylabel("Equity (normalized to 1)")
    ax_eq.set_xlabel("Date")
    ax_eq.set_title(
        f"{symbol} equity curves from true vs predicted returns"
    )
    ax_eq.legend(loc="upper left")
    ax_eq.grid(True, linestyle="--", alpha=0.3)

    outfile_eq = figures_dir / "ch15_pytorch_timeseries_regression_equity.png"
    fig_eq.tight_layout()
    fig_eq.savefig(outfile_eq, dpi=300)
    plt.close(fig_eq)


if __name__ == "__main__":
    main()
