"""Fast unit tests (no downloads, no model training beyond a few seconds)."""
import numpy as np
import pandas as pd
import pytest

from src import evaluate, features
from src.backtest import origins_for
from src.drift import cusum, page_hinkley, psi, seasonal_adjust
from src.models.baselines import SeasonalMean, SeasonalNaive
from src.models.gbm import GBMDirect


@pytest.fixture
def synthetic():
    idx = pd.date_range("2018-01-01", periods=3 * 365, freq="D")
    rng = np.random.default_rng(0)
    t = np.arange(len(idx))
    y = (25 + 8 * np.cos(2 * np.pi * t / 365.25) + 3 * (idx.dayofweek >= 5)
         + rng.normal(0, 1.5, len(idx)))
    return pd.Series(y, index=idx, name="y")


def test_calendar_features_shapes(synthetic):
    f = features.calendar_features(synthetic.index)
    assert len(f) == len(synthetic)
    assert set(["dow_sin", "doy_cos", "is_weekend", "is_holiday"]).issubset(f.columns)
    assert f["is_weekend"].sum() > 0
    assert np.allclose(f["dow_sin"] ** 2 + f["dow_cos"] ** 2, 1)


def test_fourier_terms_orthogonal_period(synthetic):
    f = features.fourier_terms(synthetic.index, period=365.25, K=2)
    assert f.shape[1] == 4
    assert f.abs().max().max() <= 1.0


def test_lag_features_use_only_past(synthetic):
    f = features.lag_features(synthetic)
    # lag_1 at t equals y at t-1
    assert np.isclose(f["lag_1"].iloc[10], synthetic.iloc[9])
    # rolling mean at t uses up to t-1 only
    assert np.isclose(f["roll_mean_7"].iloc[20], synthetic.iloc[13:20].mean())


def test_metrics_basic():
    y = np.array([10, 20, 30.0]); yhat = np.array([12, 18, 33.0])
    assert evaluate.mae(y, yhat) == pytest.approx(7 / 3)
    assert evaluate.bias(y, yhat) == pytest.approx(1.0)
    assert evaluate.mape(y, y) == 0
    assert evaluate.smape(y, y) == 0
    train = np.arange(50.0)
    assert evaluate.mase(y, yhat, train, m=7) == pytest.approx(evaluate.mae(y, yhat) / 7)


def test_business_impact_asymmetry():
    fc = pd.DataFrame({"y": [10.0, 10.0], "yhat": [12.0, 8.0]})
    r = evaluate.business_impact(fc, "uci_power")
    b = evaluate.config.BUSINESS["uci_power"]
    expected = (2 * b["cost_over_per_unit"] + 2 * b["cost_under_per_unit"]) / 2
    assert r["mean_daily_cost_per_profile"] == pytest.approx(expected)
    assert r["share_days_under_forecast"] == 0.5


def test_seasonal_naive_and_mean(synthetic):
    sn = SeasonalNaive().fit(synthetic)
    out = sn.forecast(synthetic, 10)
    assert len(out) == 10 and np.isclose(out[0], synthetic.iloc[-7])
    sm = SeasonalMean(4).fit(synthetic)
    assert len(sm.forecast(synthetic, 28)) == 28


def test_origins_spacing(synthetic):
    window = synthetic.iloc[-112:]
    o = origins_for(window, horizon=28, step=14)
    assert len(o) == 7
    assert o[-1] + pd.Timedelta(days=27) <= window.index[-1]


def test_gbm_direct_beats_naive_on_synthetic(synthetic):
    train, test = synthetic.iloc[:-56], synthetic.iloc[-56:]
    g = GBMDirect().fit(train)
    yhat = g.forecast(train, 28)
    naive = SeasonalNaive().forecast(train, 28)
    assert evaluate.mae(test.iloc[:28], yhat) < evaluate.mae(test.iloc[:28], naive)


def test_psi_detects_shift():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 2000)
    assert psi(ref, rng.normal(0, 1, 500)) < 0.1
    assert psi(ref, rng.normal(1.5, 1, 500)) > 0.25


def test_page_hinkley_and_cusum_fire_on_shift():
    rng = np.random.default_rng(2)
    x = np.concatenate([rng.normal(0, 1, 200), rng.normal(3, 1, 100)])
    ph = page_hinkley(x, delta=0.05, threshold=20)
    assert any(i >= 200 for i, _ in ph)
    assert any(i >= 200 for i in cusum(x, k=0.5, h=5))
    quiet = rng.normal(0, 1, 300)
    assert len(cusum(quiet, k=0.5, h=8)) == 0


def test_seasonal_adjust_removes_profile(synthetic):
    adj = seasonal_adjust(synthetic)
    assert adj.std() < synthetic.std() * 0.5
    assert adj.groupby(adj.index.dayofweek).mean().std() < 0.5
