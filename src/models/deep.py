"""LSTM / GRU sequence-to-vector forecasters (PyTorch, CPU friendly).

Architecture
------------
encoder   : RNN over the last LOOKBACK days of [scaled target + calendar features]
decoder   : the final hidden state is concatenated with the *known future*
            calendar features of the HORIZON target days and passed through an
            MLP that emits all HORIZON steps at once (direct multi-output).
Training  : MSE on standardised target, Adam, early stopping on a validation
            window, fixed seeds for reproducibility.
"""
from __future__ import annotations

import copy
import random

import numpy as np
import pandas as pd
import torch
from torch import nn

from .. import config
from ..features import calendar_features
from .base import Forecaster, future_index

CAL_COLS = ["dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_weekend", "is_holiday"]


def _seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class _Seq2Vec(nn.Module):
    def __init__(self, n_in, n_future_feats, horizon, hidden=64, layers=1, dropout=0.1, cell="lstm"):
        super().__init__()
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(n_in, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_future_feats * horizon, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, horizon),
        )

    def forward(self, x_past, x_future):
        out, _ = self.rnn(x_past)
        h = out[:, -1, :]
        z = torch.cat([h, x_future.flatten(1)], dim=1)
        return self.head(z)


class RNNForecaster(Forecaster):
    def __init__(self, cell="lstm", lookback=config.LOOKBACK, horizon=config.HORIZON,
                 hidden=64, layers=1, dropout=0.1, lr=1e-3, epochs=200, patience=15,
                 batch_size=32, seed=config.RANDOM_SEED, country="FR"):
        self.cell, self.lookback, self.horizon = cell, lookback, horizon
        self.hidden, self.layers, self.dropout = hidden, layers, dropout
        self.lr, self.epochs, self.patience, self.batch_size = lr, epochs, patience, batch_size
        self.seed, self.country = seed, country
        self.name = cell.upper()
        self.model_ = None
        self.mu_ = self.sd_ = None
        self.history_: list[dict] = []
        self.val_series_: pd.Series | None = None

    # ------------------------------------------------------------- data prep
    def _cal(self, index):
        return calendar_features(index, self.country)[CAL_COLS].to_numpy(np.float32)

    def _windows(self, y: pd.Series, cal: np.ndarray, start: int, end: int):
        """Build (x_past, x_future, target) for every origin t in [start, end)
        such that t-lookback >= 0 and t+horizon <= len(y)."""
        ys = ((y.values - self.mu_) / self.sd_).astype(np.float32)
        Xp, Xf, Y = [], [], []
        for t in range(max(start, self.lookback), min(end, len(y) - self.horizon + 1)):
            past = np.column_stack([ys[t - self.lookback:t], cal[t - self.lookback:t]])
            Xp.append(past)
            Xf.append(cal[t:t + self.horizon])
            Y.append(ys[t:t + self.horizon])
        return (torch.tensor(np.array(Xp)), torch.tensor(np.array(Xf)), torch.tensor(np.array(Y)))

    # --------------------------------------------------------------------- fit
    def fit(self, y: pd.Series, y_val: pd.Series | None = None):
        """`y` is the training series; `y_val` (optional) directly follows it and
        is used only for early stopping."""
        _seed(self.seed)
        self.mu_, self.sd_ = float(y.mean()), float(y.std())
        full = y if y_val is None else pd.concat([y, y_val])
        cal = self._cal(full.index)
        n_train = len(y)
        Xp, Xf, Y = self._windows(full, cal, 0, n_train - self.horizon + 1)
        if y_val is not None:
            Xpv, Xfv, Yv = self._windows(full, cal, n_train - self.horizon + 1, len(full))
        else:
            Xpv = None
        self.model_ = _Seq2Vec(Xp.shape[-1], Xf.shape[-1], self.horizon, self.hidden,
                               self.layers, self.dropout, self.cell)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
        loss_fn = nn.MSELoss()
        best, best_state, bad = np.inf, None, 0
        n = len(Xp)
        for epoch in range(self.epochs):
            self.model_.train()
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                pred = self.model_(Xp[idx], Xf[idx])
                loss = loss_fn(pred, Y[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                opt.step()
                tot += loss.item() * len(idx)
            train_loss = tot / n
            if Xpv is not None:
                self.model_.eval()
                with torch.no_grad():
                    val_loss = loss_fn(self.model_(Xpv, Xfv), Yv).item()
            else:
                val_loss = train_loss
            sched.step(val_loss)
            self.history_.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            if val_loss < best - 1e-5:
                best, best_state, bad = val_loss, copy.deepcopy(self.model_.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.best_val_loss_ = best
        self.epochs_run_ = len(self.history_)
        return self

    # ---------------------------------------------------------------- forecast
    def forecast(self, history: pd.Series, horizon: int):
        assert horizon <= self.horizon
        fidx = future_index(history, self.horizon)
        past_idx = history.index[-self.lookback:]
        ys = ((history.values[-self.lookback:] - self.mu_) / self.sd_).astype(np.float32)
        xp = np.column_stack([ys, self._cal(past_idx)])[None]
        xf = self._cal(fidx)[None]
        with torch.no_grad():
            out = self.model_(torch.tensor(xp), torch.tensor(xf)).numpy()[0]
        return (out * self.sd_ + self.mu_)[:horizon]
