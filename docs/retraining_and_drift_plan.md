# Retraining and Drift Management Plan

This document turns the monitors implemented in `src/drift.py` into an operating procedure for
the daily demand forecast. It assumes the forecast is produced every day for the next 28 days and
consumed by a day-ahead procurement (or replenishment) process.

## 1. What can go wrong

| failure mode | example in this dataset | symptom | detector |
|---|---|---|---|
| **Data quality drift** | meter outage (22 days with < 90 % of minute readings) | implausible drops, missing days | ingestion checks (coverage, range, schema) |
| **Input / covariate drift** | unusually cold winter, heat-wave | target distribution shifts in a way the seasonal profile does not explain | PSI / KS on seasonally adjusted demand |
| **Concept drift** | new appliance, household size changes, tariff change, store remodel | persistent bias in residuals; weekly profile reshapes | Page-Hinkley / CUSUM on residuals, rolling bias |
| **Performance decay** | gradual -3.7 %/year trend not captured by a stale model | rolling MAE creeps above the validated level | rolling MAE vs. tolerance band |
| **Regime events** | vacation weeks at ~9 kWh/day (Aug 2008, Aug 2010) | all models over-forecast for 1-3 weeks | anomaly flags + exogenous calendar |

## 2. Monitoring stack

All metrics are computed daily once actuals arrive and written to a monitoring table
(`reports/metrics/drift_performance_monitor.csv` shows the format).

### 2.1 Data checks (before forecasting)
* Schema and freshness: all expected days present, latest day <= 1 day old.
* Coverage: fraction of valid minute readings >= 90 %; otherwise the day is flagged and imputed,
  and imputed days are excluded from residual statistics.
* Range: value within [0, 3 x 99th percentile of training data].
* Anomaly flag: robust z-score of the STL remainder > 4 -> stored, not removed (a real spike is still demand).

### 2.2 Distribution drift (weekly)
* Compare the last 90 days with the same 90-day window one year earlier **after seasonal
  adjustment** (raw PSI between seasons is always large and meaningless).
* PSI < 0.10 stable, 0.10-0.25 moderate (investigate), > 0.25 severe (retrain).
* KS test p-value < 0.01 as a second opinion; both must agree before an automatic action.
* Also track the mean and standard deviation of the weekly profile (Mon..Sun means) and the
  weekend/weekday ratio.

### 2.3 Model drift (daily)
Residuals use the forecast with the shortest live horizon (h <= 14 given the 14-day re-forecast
cadence in the backtest; h = 1 in production where forecasts are issued daily).

| monitor | statistic | alarm rule |
|---|---|---|
| rolling accuracy | 28-day rolling MAE | > 1.25 x validation MAE for 7 consecutive days |
| rolling bias | 28-day mean residual | |bias| > 0.5 x validation MAE for 14 consecutive days |
| sequential shift | Page-Hinkley (delta = 0.05 MAE, lambda = 8 MAE) | any alarm |
| sequential shift | CUSUM on standardised residuals (k = 0.5, h = 5) | any alarm |
| calibration | 90 % interval coverage over last 56 days | coverage < 80 % or > 97 % |
| business | asymmetric cost per day vs. budget | 28-day cost > 1.3 x planned |

### 2.4 Champion / challenger
Every retraining produces challengers. The 3 best non-baseline models on the validation window
plus their mean ensemble are always scored; the champion is replaced only if the challenger wins
on **both** MAE and asymmetric cost over the full backtest window (not on a single week).

## 3. Retraining schedule

| model | why | scheduled refit | cost |
|---|---|---|---|
| SARIMAX + Fourier | parameters are cheap to re-estimate; state is updated daily via the Kalman filter without refit | re-estimate **weekly**; order re-selected **quarterly** | ~5 s refit, ~5 min order search |
| Prophet | curve fit, benefits from seeing recent changepoints | refit **daily** at forecast time (as done in the backtest) | ~2 s |
| GBM (features) | tree model, deterministic | refit **weekly**; hyper-parameters quarterly | ~3 s |
| LSTM / GRU | slowest, most variance between runs | **monthly** full retrain with early stopping on the last 112 days; weekly warm-start fine-tune (5 epochs) | ~1-3 min CPU |
| Ensemble weights | | re-derived at every monthly retrain from the validation window | negligible |

Scheduled retraining uses a sliding window: all history, but with the yearly Fourier/Prophet
seasonality fitted on at least 3 full years, and the validation window always the 112 most
recent days.

## 4. Triggered retraining

Retraining is triggered **outside the schedule** when any of the following occurs:

1. Severe distribution drift: seasonally adjusted PSI > 0.25 **and** KS p < 0.01.
2. Rolling MAE alarm (section 2.3) for 7 consecutive days.
3. Two or more Page-Hinkley / CUSUM alarms within 28 days.
4. A known structural change is announced (new appliance / store opening / price change).
   The event date is added to the holiday-style regressor table so Prophet and the GBM learn a
   level shift rather than treating it as noise.

Triggered retraining procedure:

1. Re-run data checks; quarantine anomalous days that are explained by outages.
2. Refit all models on data up to yesterday, re-tune on the last 112 days.
3. Backtest with rolling origins exactly as in `run_pipeline.py`; compare with the champion.
4. Gate: promote only if MAE improves by > 2 % **and** asymmetric cost does not worsen.
5. Record the version, data window, hyper-parameters, metrics and the triggering alarm in
   `models_store/registry.jsonl`; keep the previous champion for instant rollback.
6. Reset the sequential detectors (Page-Hinkley, CUSUM) after promotion.

## 5. What the backtest in this repository showed

The drift section of `reports/REPORT.md` reports the actual values. Reading them:

* Seasonally adjusted PSI between the last training year and the hold-out window was moderate,
  consistent with the hold-out autumn being slightly different from the previous one but not a
  new regime; scheduled (not triggered) retraining is appropriate.
* The rolling MAE monitor would have raised an alarm in the second half of October 2010, when
  consumption volatility rose with the start of the heating season; a weekly SARIMAX / GBM refit
  and the daily Prophet refit are exactly the mechanism that absorbs this.
* Page-Hinkley alarms cluster around the mid-October spike and dip, an occupancy change rather
  than a true drift; the "two alarms within 28 days" rule would still have triggered a retrain,
  which is acceptable because retraining is cheap, whereas missing a real shift is not.

## 6. Ownership and cadence summary

| cadence | action |
|---|---|
| daily | ingest actuals, data checks, forecast (Prophet refit, SARIMAX filter update), residual monitors, dashboard refresh |
| weekly | SARIMAX + GBM refit, PSI/KS drift check, interval-coverage check |
| monthly | LSTM/GRU retrain, ensemble re-weighting, champion/challenger review |
| quarterly | SARIMA order and GBM hyper-parameter re-selection, review of cost assumptions in `src/config.py` |
| on alarm | triggered retraining procedure (section 4) |
