"""Time-based feature engineering for daily demand series."""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import holidays as _holidays
except ImportError:  # pragma: no cover
    _holidays = None


def calendar_features(index: pd.DatetimeIndex, country: str = "FR") -> pd.DataFrame:
    """Deterministic calendar features that are known for any future date."""
    idx = pd.DatetimeIndex(index)
    f = pd.DataFrame(index=idx)
    f["dayofweek"] = idx.dayofweek
    f["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    f["month"] = idx.month
    f["dayofyear"] = idx.dayofyear
    f["weekofyear"] = idx.isocalendar().week.astype(int).values
    f["quarter"] = idx.quarter
    f["year"] = idx.year
    f["days_since_start"] = (idx - idx[0]).days
    # cyclical encodings
    f["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    f["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    f["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    f["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    f["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    f["is_holiday"] = 0
    if _holidays is not None:
        try:
            hol = _holidays.country_holidays(country, years=sorted(set(idx.year)))
            f["is_holiday"] = np.array([int(d in hol) for d in idx.date])
        except Exception:
            pass
    return f


def fourier_terms(index: pd.DatetimeIndex, period: float = 365.25, K: int = 3,
                  origin: pd.Timestamp | None = None) -> pd.DataFrame:
    """Fourier basis for long seasonalities (used as SARIMAX exogenous regressors)."""
    idx = pd.DatetimeIndex(index)
    origin = origin if origin is not None else idx[0]
    t = (idx - origin).days.values.astype(float)
    cols = {}
    for k in range(1, K + 1):
        cols[f"sin{k}_{int(period)}"] = np.sin(2 * np.pi * k * t / period)
        cols[f"cos{k}_{int(period)}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(cols, index=idx)


def lag_features(y: pd.Series, lags=(1, 2, 3, 7, 14, 21, 28, 364),
                 windows=(7, 14, 28)) -> pd.DataFrame:
    """Lagged and rolling statistics of the target (uses only past values)."""
    f = pd.DataFrame(index=y.index)
    for lag in lags:
        f[f"lag_{lag}"] = y.shift(lag)
    shifted = y.shift(1)
    for w in windows:
        f[f"roll_mean_{w}"] = shifted.rolling(w).mean()
        f[f"roll_std_{w}"] = shifted.rolling(w).std()
        f[f"roll_min_{w}"] = shifted.rolling(w).min()
        f[f"roll_max_{w}"] = shifted.rolling(w).max()
    f["ewm_7"] = shifted.ewm(span=7).mean()
    f["ewm_28"] = shifted.ewm(span=28).mean()
    f["diff_7"] = y.shift(1) - y.shift(8)
    return f


def build_feature_frame(y: pd.Series, country: str = "FR") -> pd.DataFrame:
    """Full supervised-learning frame (calendar + lags). Rows with NaN lags dropped."""
    f = pd.concat([calendar_features(y.index, country), lag_features(y)], axis=1)
    f["y"] = y.values
    return f.dropna()
