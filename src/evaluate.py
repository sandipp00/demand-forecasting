"""Forecast accuracy metrics and translation of error into business cost."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
# Accuracy metrics
# --------------------------------------------------------------------------- #
def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mape(y, yhat, eps=1e-9):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(np.abs((y - yhat) / np.maximum(np.abs(y), eps))) * 100)


def smape(y, yhat, eps=1e-9):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = np.maximum((np.abs(y) + np.abs(yhat)) / 2, eps)
    return float(np.mean(np.abs(y - yhat) / denom) * 100)


def bias(y, yhat):
    """Mean error (forecast - actual). Positive = over-forecast."""
    return float(np.mean(np.asarray(yhat) - np.asarray(y)))


def mase(y, yhat, y_train, m: int = config.SEASONAL_PERIOD):
    """Mean Absolute Scaled Error relative to in-sample seasonal-naive forecast."""
    y_train = np.asarray(y_train, float)
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return mae(y, yhat) / scale


def all_metrics(y, yhat, y_train) -> dict:
    return {
        "MAE": mae(y, yhat),
        "RMSE": rmse(y, yhat),
        "MAPE_%": mape(y, yhat),
        "sMAPE_%": smape(y, yhat),
        "MASE": mase(y, yhat, y_train),
        "Bias": bias(y, yhat),
    }


def metrics_by_horizon(fc: pd.DataFrame) -> pd.DataFrame:
    """fc columns: model, origin, date, h, y, yhat  -> MAE per model per horizon step."""
    g = fc.assign(ae=(fc["y"] - fc["yhat"]).abs()).groupby(["model", "h"])["ae"].mean()
    return g.unstack("model")


# --------------------------------------------------------------------------- #
# Business impact
# --------------------------------------------------------------------------- #
def business_impact(fc_model: pd.DataFrame, dataset: str | None = None) -> dict:
    """Translate the forecast errors of one model into money.

    Newsvendor-style asymmetric cost:
      cost = c_over * max(yhat - y, 0) + c_under * max(y - yhat, 0)
    scaled from one profile to a portfolio.  Also computes the safety buffer
    required to reach the target service level (quantile of under-forecast).
    """
    dataset = dataset or config.DATASET
    b = config.BUSINESS[dataset]
    err = (fc_model["yhat"] - fc_model["y"]).to_numpy()
    over = np.clip(err, 0, None)
    under = np.clip(-err, 0, None)
    daily_cost_profile = b["cost_over_per_unit"] * over + b["cost_under_per_unit"] * under
    n_days = len(err)
    mean_daily_cost = float(daily_cost_profile.mean())
    # buffer that covers under-forecast on `service_level` of days
    buffer = float(np.quantile(under, b["service_level"]))
    return {
        "n_forecast_days": n_days,
        "mean_abs_error": float(np.abs(err).mean()),
        "share_days_under_forecast": float((err < 0).mean()),
        "mean_daily_cost_per_profile": mean_daily_cost,
        "annual_cost_portfolio": mean_daily_cost * 365 * b["portfolio_size"],
        f"safety_buffer_p{int(b['service_level']*100)}": buffer,
        "annual_buffer_cost_portfolio": buffer * b["cost_over_per_unit"] * 365 * b["portfolio_size"],
        "currency": b["currency"],
        "portfolio_size": b["portfolio_size"],
    }


def business_table(fc: pd.DataFrame, dataset: str | None = None) -> pd.DataFrame:
    rows = {}
    for model, g in fc.groupby("model"):
        rows[model] = business_impact(g, dataset)
    return pd.DataFrame(rows).T.sort_values("annual_cost_portfolio")
