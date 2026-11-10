"""SARIMA / SARIMAX with Fourier regressors for the yearly cycle.

Weekly seasonality is modelled by the seasonal ARIMA part (m=7). The yearly
cycle (period 365.25) is far too long for a seasonal ARIMA term, so it enters as
a small set of Fourier harmonics in the exogenous matrix (the "dynamic harmonic
regression" approach of Hyndman & Athanasopoulos).
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .. import config
from ..features import fourier_terms
from .base import Forecaster, future_index


class SarimaFourier(Forecaster):
    def __init__(self, order=None, seasonal_order=None, K: int = 3, use_fourier: bool = True,
                 search: bool = True):
        self.order = order
        self.seasonal_order = seasonal_order
        self.K = K
        self.use_fourier = use_fourier
        self.search = search and order is None
        self.res_ = None
        self.origin_ = None
        self.search_log_: pd.DataFrame | None = None
        self.name = "SARIMA" if not use_fourier else "SARIMAX+Fourier"

    # ------------------------------------------------------------------ helpers
    def _exog(self, index):
        if not self.use_fourier:
            return None
        return fourier_terms(index, config.YEARLY_PERIOD, self.K, origin=self.origin_)

    def _fit_one(self, y, order, seasonal_order):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = SARIMAX(y, exog=self._exog(y.index), order=order,
                          seasonal_order=seasonal_order,
                          enforce_stationarity=False, enforce_invertibility=False)
            return mod.fit(disp=False, maxiter=200)

    # --------------------------------------------------------------------- API
    def fit(self, y: pd.Series):
        self.origin_ = y.index[0]
        if self.search:
            grid = itertools.product([0, 1, 2], [0, 1], [0, 1, 2], [0, 1], [0, 1], [0, 1])
            rows = []
            best = None
            for p, d, q, P, D, Q in grid:
                if p == q == 0 and P == Q == 0:
                    continue
                try:
                    res = self._fit_one(y, (p, d, q), (P, D, Q, config.SEASONAL_PERIOD))
                except Exception:
                    continue
                rows.append({"order": (p, d, q), "seasonal_order": (P, D, Q, 7),
                             "aic": res.aic, "bic": res.bic})
                if best is None or res.aic < best[0]:
                    best = (res.aic, (p, d, q), (P, D, Q, config.SEASONAL_PERIOD), res)
            self.search_log_ = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
            _, self.order, self.seasonal_order, self.res_ = best
        else:
            self.res_ = self._fit_one(y, self.order, self.seasonal_order)
        self.name = (f"SARIMAX{self.order}x{self.seasonal_order}+F{self.K}" if self.use_fourier
                     else f"SARIMA{self.order}x{self.seasonal_order}")
        return self

    def _applied(self, history: pd.Series):
        """Re-apply the fitted parameters to a (longer) history without re-estimating."""
        if len(history) == self.res_.nobs and history.index[-1] == self.res_.data.row_labels[-1]:
            return self.res_
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.res_.apply(history, exog=self._exog(history.index))

    def forecast(self, history, horizon):
        res = self._applied(history)
        fidx = future_index(history, horizon)
        fc = res.get_forecast(steps=horizon, exog=self._exog(fidx))
        return np.asarray(fc.predicted_mean)

    def forecast_intervals(self, history, horizon, alpha=0.1):
        res = self._applied(history)
        fidx = future_index(history, horizon)
        fc = res.get_forecast(steps=horizon, exog=self._exog(fidx))
        ci = fc.conf_int(alpha=alpha)
        return ci.iloc[:, 0].to_numpy(), ci.iloc[:, 1].to_numpy()

    def summary_dict(self):
        return {"order": list(self.order), "seasonal_order": list(self.seasonal_order),
                "fourier_K": self.K if self.use_fourier else 0,
                "aic": float(self.res_.aic), "bic": float(self.res_.bic)}
