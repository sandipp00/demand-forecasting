"""Facebook/Meta Prophet with weekly + yearly seasonality and national holidays."""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from .base import Forecaster, future_index

logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet").setLevel(logging.ERROR)


class ProphetModel(Forecaster):
    def __init__(self, changepoint_prior_scale=0.05, seasonality_prior_scale=10.0,
                 seasonality_mode="additive", country="FR", yearly_fourier=10,
                 refit_each_origin=True):
        self.cps = changepoint_prior_scale
        self.sps = seasonality_prior_scale
        self.mode = seasonality_mode
        self.country = country
        self.yearly_fourier = yearly_fourier
        self.refit_each_origin = refit_each_origin
        self.model_ = None
        self.fitted_end_ = None
        self.name = f"Prophet(cps={changepoint_prior_scale},sps={seasonality_prior_scale},{seasonality_mode[:3]})"

    def _new(self):
        from prophet import Prophet
        m = Prophet(changepoint_prior_scale=self.cps, seasonality_prior_scale=self.sps,
                    seasonality_mode=self.mode, yearly_seasonality=self.yearly_fourier,
                    weekly_seasonality=True, daily_seasonality=False,
                    interval_width=0.9, uncertainty_samples=200)
        if self.country:
            try:
                m.add_country_holidays(country_name=self.country)
            except Exception:
                pass
        return m

    def _fit_on(self, y: pd.Series):
        df = pd.DataFrame({"ds": y.index, "y": y.values})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_ = self._new().fit(df)
        self.fitted_end_ = y.index[-1]
        return self

    def fit(self, y):
        return self._fit_on(y)

    def _predict_df(self, history, horizon):
        if self.refit_each_origin and history.index[-1] != self.fitted_end_:
            self._fit_on(history)
        fidx = future_index(history, horizon)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.model_.predict(pd.DataFrame({"ds": fidx}))

    def forecast(self, history, horizon):
        return self._predict_df(history, horizon)["yhat"].to_numpy()

    def forecast_intervals(self, history, horizon, alpha=0.1):
        p = self._predict_df(history, horizon)
        return p["yhat_lower"].to_numpy(), p["yhat_upper"].to_numpy()
