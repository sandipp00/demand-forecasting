"""Common forecaster interface used by the backtesting harness."""
from __future__ import annotations

import numpy as np
import pandas as pd


class Forecaster:
    """Minimal interface.

    fit(y)                  : estimate parameters on a training series.
    forecast(history, h)    : return `h` point forecasts for the days following
                              `history` (history = all actuals up to the origin;
                              it may be longer than the series used in fit()).
    """

    name: str = "base"

    def fit(self, y: pd.Series) -> "Forecaster":
        raise NotImplementedError

    def forecast(self, history: pd.Series, horizon: int) -> np.ndarray:
        raise NotImplementedError

    def forecast_intervals(self, history: pd.Series, horizon: int, alpha: float = 0.1):
        """Optional (lower, upper) arrays; default None."""
        return None


def future_index(history: pd.Series, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
