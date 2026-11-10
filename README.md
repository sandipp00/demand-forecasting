# Demand Forecasting / Time Series Modeling

End-to-end forecasting system for daily consumption that explores trend, seasonality and
anomalies, engineers time-based features, compares SARIMA, Prophet, LSTM/GRU and a
feature-based gradient boosting model under a rolling-origin backtest, translates error into
money, and ships a drift-monitoring and retraining plan.

* **Results**: [`reports/REPORT.md`](reports/REPORT.md) (figures in `reports/figures/`, numbers in `reports/metrics/`)
* **Retraining and drift plan**: [`docs/retraining_and_drift_plan.md`](docs/retraining_and_drift_plan.md)
* **Walk-through notebook**: [`notebooks/demand_forecasting_walkthrough.ipynb`](notebooks/demand_forecasting_walkthrough.ipynb)

## Dataset

Primary: **UCI Individual Household Electric Power Consumption**
(https://archive.ics.uci.edu/dataset/235). 2,075,259 one-minute readings from a house near Paris,
Dec 2006 to Nov 2010, aggregated to 1,440 daily kWh values. Downloaded automatically on first run.

Also supported: **Kaggle M5 Forecasting - Accuracy** (Walmart unit sales). Kaggle requires a
login, so the files are not fetched automatically. To use it, place `sales_train_evaluation.csv`
and `calendar.csv` in `data/raw/m5/` and set `DATASET = "m5"` in `src/config.py`; the loader
aggregates to total daily units and the rest of the pipeline runs unchanged.

## Quick start

```bash
pip install -r requirements.txt
python run_pipeline.py            # full run, ~10-15 min on CPU
python run_pipeline.py --fast     # 2-3 min smoke run (no grid searches)
python -m pytest tests -q         # unit tests
```

## What the pipeline does

| stage | module | output |
|---|---|---|
| load + clean | `src/data.py` | daily series, coverage-based imputation of 22 low-quality days |
| EDA | `src/eda.py` | trend, weekly/yearly profiles, MSTL decomposition, ACF/PACF, ADF/KPSS, robust anomaly flags, heat-map |
| features | `src/features.py` | calendar, cyclical encodings, FR holidays, Fourier harmonics, lags, rolling stats |
| models | `src/models/` | seasonal-naive baselines, SARIMA, SARIMAX + Fourier, Prophet, GBM (direct), LSTM, GRU, ensemble |
| backtest | `src/backtest.py` | 28-day forecasts from rolling origins every 14 days in validation and hold-out windows |
| evaluation | `src/evaluate.py` | MAE, RMSE, MAPE, sMAPE, MASE, bias, error by horizon, asymmetric cost, safety buffer / service level |
| drift | `src/drift.py` | PSI, KS, Page-Hinkley, CUSUM, rolling-MAE monitor, simulated over the hold-out |
| report | `src/report.py` | `reports/REPORT.md` |

### Evaluation protocol

Chronological split: train (1,216 days) / validation (112 days) / test (112 days). All tuning
(SARIMA order by AIC, Prophet priors, RNN size and early stopping) happens on validation. Models
are refit on train+validation and produce 28-day forecasts from 7 rolling origins in the hold-out
window (196 forecast points per model). Prophet refits at every origin; SARIMAX re-applies its
parameters to the extended history through the Kalman filter.

### Business impact

`src/config.py` holds editable cost assumptions: an energy retailer buying for 100,000 households
pays EUR 0.04/kWh for over-procurement and EUR 0.12/kWh for shortfalls. The report converts each
model's errors into an annual cost, and sizes the safety buffer needed for a 95 % service level.

## Repository layout

```
run_pipeline.py              orchestrates everything
src/                         library code (see table above)
src/models/                  one file per model family, common Forecaster interface
reports/REPORT.md            generated results report
reports/figures/             13 figures
reports/metrics/             CSV/JSON: metrics, forecasts, grids, drift report
docs/                        retraining and drift management plan
notebooks/                   walk-through notebook built on the saved outputs
tests/                       unit tests for features, metrics, drift detectors, baselines
data/raw, data/processed     downloaded and cached data (git-ignored)
```

## Reproducibility

Seeds are fixed (`RANDOM_SEED = 42`); PyTorch runs on CPU. Package versions are pinned in
`requirements.txt`. All intermediate artefacts (forecast frames, grid results, timings) are
written to `reports/metrics/` so any table in the report can be re-derived.
