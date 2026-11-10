"""Gradient-boosted trees on engineered time features (direct multi-step).

Every lag feature is shifted by at least ``HORIZON`` days, so one model can
produce all 28 daily forecasts straight from information available at the
forecast origin: no recursion and therefore no compounding of one-step errors.
It demonstrates how the hand-built calendar + lag features translate into a
competitive tabular forecaster and gives the sequence models a strong benchmark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .. import config
from ..features import calendar_features
from .base import Forecaster, future_index

DROP = {"year", "days_since_start", "dayofyear", "weekofyear"}  # avoid extrapolation traps


def direct_lag_features(y: pd.Series, min_lag: int) -> pd.DataFrame:
    """Lags / rolling stats that are all >= min_lag days old."""
    f = pd.DataFrame(index=y.index)
    for lag in (0, 1, 2, 3, 7, 14, 21, 28):
        f[f"lag_{min_lag + lag}"] = y.shift(min_lag + lag)
    f["lag_364"] = y.shift(364)
    f["lag_371"] = y.shift(371)
    base = y.shift(min_lag)
    for w in (7, 14, 28, 56):
        f[f"roll_mean_{w}"] = base.rolling(w).mean()
        f[f"roll_std_{w}"] = base.rolling(w).std()
    f["roll_min_28"] = base.rolling(28).min()
    f["roll_max_28"] = base.rolling(28).max()
    f["ewm_14"] = base.ewm(span=14).mean()
    f["yoy_level_ratio"] = base.rolling(28).mean() / y.shift(364).rolling(28).mean()
    return f


class GBMDirect(Forecaster):
    name = "GBM(features,direct)"

    def __init__(self, horizon=config.HORIZON, country="FR", **params):
        self.horizon = horizon
        self.country = country
        self.params = dict(max_iter=500, learning_rate=0.04, max_leaf_nodes=15,
                           min_samples_leaf=20, l2_regularization=1.0, random_state=42)
        self.params.update(params)
        self.model_ = None
        self.cols_ = None

    def _frame(self, y: pd.Series):
        f = pd.concat([calendar_features(y.index, self.country),
                       direct_lag_features(y, self.horizon)], axis=1)
        return f.drop(columns=[c for c in DROP if c in f.columns])

    def fit(self, y: pd.Series):
        f = self._frame(y)
        f["y"] = y.values
        f = f.dropna()
        self.cols_ = [c for c in f.columns if c != "y"]
        self.model_ = HistGradientBoostingRegressor(**self.params).fit(f[self.cols_], f["y"])
        return self

    def forecast(self, history: pd.Series, horizon: int):
        fidx = future_index(history, horizon)
        ext = pd.concat([history, pd.Series(np.nan, index=fidx)])
        f = self._frame(ext).loc[fidx, self.cols_]
        return self.model_.predict(f)

    def feature_importance(self, y: pd.Series, n_repeats=5) -> pd.Series:
        from sklearn.inspection import permutation_importance
        f = self._frame(y); f["y"] = y.values; f = f.dropna()
        r = permutation_importance(self.model_, f[self.cols_], f["y"], n_repeats=n_repeats,
                                   random_state=0)
        return pd.Series(r.importances_mean, index=self.cols_).sort_values(ascending=False)
