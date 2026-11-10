"""Rolling-origin evaluation harness.

For an evaluation window [start, end] we place forecast origins every
``ORIGIN_STEP`` days and ask each model for ``HORIZON`` daily forecasts from
each origin, using all actuals up to that origin as history. The result is a
tidy frame with one row per (model, origin, horizon step).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config
from .models.base import Forecaster


def origins_for(window: pd.Series, horizon=config.HORIZON, step=config.ORIGIN_STEP):
    """Forecast origins (= first forecast date) inside `window`."""
    out = []
    d = window.index[0]
    while d + pd.Timedelta(days=horizon - 1) <= window.index[-1]:
        out.append(d)
        d += pd.Timedelta(days=step)
    return out


def rolling_forecasts(models: list[Forecaster], y: pd.Series, window: pd.Series,
                      horizon=config.HORIZON, step=config.ORIGIN_STEP,
                      with_intervals=False, verbose=True) -> pd.DataFrame:
    rows = []
    timings = {}
    for m in models:
        t0 = time.time()
        for origin in origins_for(window, horizon, step):
            history = y.loc[: origin - pd.Timedelta(days=1)]
            fidx = pd.date_range(origin, periods=horizon, freq="D")
            yhat = np.asarray(m.forecast(history, horizon), float)
            lo = hi = [np.nan] * horizon
            if with_intervals:
                ci = m.forecast_intervals(history, horizon)
                if ci is not None:
                    lo, hi = ci
            for h in range(horizon):
                rows.append({"model": m.name, "origin": origin, "date": fidx[h], "h": h + 1,
                             "y": float(y.loc[fidx[h]]), "yhat": float(yhat[h]),
                             "lo": float(lo[h]), "hi": float(hi[h])})
        timings[m.name] = time.time() - t0
        if verbose:
            print(f"  {m.name:<40s} forecasted in {timings[m.name]:6.1f}s")
    fc = pd.DataFrame(rows)
    fc.attrs["timings"] = timings
    return fc


def add_ensemble(fc: pd.DataFrame, members: list[str], name="Ensemble(mean)") -> pd.DataFrame:
    sub = fc[fc["model"].isin(members)]
    ens = (sub.groupby(["origin", "date", "h"], as_index=False)
              .agg(y=("y", "first"), yhat=("yhat", "mean")))
    ens["model"] = name
    ens["lo"] = np.nan
    ens["hi"] = np.nan
    return pd.concat([fc, ens[fc.columns]], ignore_index=True)


def summarise(fc: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    from .evaluate import all_metrics
    rows = {}
    for model, g in fc.groupby("model"):
        rows[model] = all_metrics(g["y"], g["yhat"], y_train)
        rows[model]["n_points"] = len(g)
    return pd.DataFrame(rows).T.sort_values("MAE")
