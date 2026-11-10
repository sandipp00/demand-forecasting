"""Naive reference models every serious model must beat."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Forecaster


class SeasonalNaive(Forecaster):
    """Repeat the value observed `m` days earlier (weekly pattern)."""

    def __init__(self, m: int = 7):
        self.m = m
        self.name = f"SeasonalNaive({m})"

    def fit(self, y):
        return self

    def forecast(self, history, horizon):
        last = history.values[-self.m:]
        reps = int(np.ceil(horizon / self.m))
        return np.tile(last, reps)[:horizon]


class SeasonalMean(Forecaster):
    """Average of the last `k` weeks for each weekday (smoother seasonal naive)."""

    def __init__(self, weeks: int = 4):
        self.weeks = weeks
        self.name = f"SeasonalMean({weeks}w)"

    def fit(self, y):
        return self

    def forecast(self, history, horizon):
        last = history.values[-7 * self.weeks:].reshape(self.weeks, 7)
        prof = last.mean(axis=0)
        reps = int(np.ceil(horizon / 7))
        return np.tile(prof, reps)[:horizon]


class LastYearNaive(Forecaster):
    """Same day last year (364 days back keeps the weekday aligned), scaled by the
    ratio of the recent 28-day level to the level a year ago."""

    name = "LastYearScaled"

    def fit(self, y):
        return self

    def forecast(self, history, horizon):
        v = history.values
        if len(v) < 364 + 28:
            return SeasonalNaive().forecast(history, horizon)
        recent = v[-28:].mean()
        year_ago_level = v[-364 - 28:-364].mean()
        scale = recent / year_ago_level if year_ago_level > 0 else 1.0
        base = v[-364:-364 + horizon]
        if len(base) < horizon:  # horizon beyond available data
            base = np.concatenate([base, np.tile(v[-7:], 10)])[:horizon]
        return base * scale
