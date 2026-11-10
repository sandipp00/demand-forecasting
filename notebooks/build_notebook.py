"""Generate the walk-through notebook (reads artefacts produced by run_pipeline.py)."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Demand Forecasting walk-through

Daily household electricity consumption (UCI), 2006-2010. This notebook walks through the
pipeline in `run_pipeline.py` using the artefacts it saved under `reports/`, and re-runs the
light-weight pieces live (feature construction, a SARIMAX fit, Prophet components, drift checks).

Sections: 1. data, 2. trend / seasonality / anomalies, 3. time features, 4. models and
backtest results, 5. error by horizon and business impact, 6. drift and retraining.""")

code("""import sys, json, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image, display, Markdown
from src import config
from src.data import load_daily, target_series, train_val_test_split
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
M = ROOT / "reports" / "metrics"; F = ROOT / "reports" / "figures"
def fig(name): display(Image(filename=str(F / name)))""")

md("## 1. Data: from 2 million minute readings to 1,440 daily values")
code("""df = load_daily(); y = target_series()
train, val, test = train_val_test_split(y)
print(f"{len(y)} days, {y.index[0].date()} -> {y.index[-1].date()}  | imputed days: {int(df.was_imputed.sum())}")
print(f"train {len(train)} | val {len(val)} | test {len(test)}")
df.head()""")
code("""y.describe().round(2).to_frame().T""")

md("""## 2. Trend, seasonality and anomalies

* a slow decline of roughly 3-4 % per year;
* a dominant yearly cycle (winter heating about doubles summer consumption);
* a weekly cycle (weekends about 4-5 kWh higher);
* holiday-period spikes and vacation dips flagged by robust z-scores on the STL remainder.""")
code("""eda = json.load(open(M / "eda_summary.json"))
keys = ["mean","std","weekday_mean","weekend_mean","winter_mean(DJF)","summer_mean(JJA)","yoy_trend_%_per_year","n_anomalies"]
display(pd.Series({k: eda[k] for k in keys}).round(2).to_frame("value"))
display(pd.DataFrame(eda["seasonal_strength"], index=["strength"]).round(3))
print(eda["stationarity"]["interpretation"])""")
code("""fig("01_series_trend.png"); fig("02_seasonality_profiles.png")""")
code("""fig("03_mstl_decomposition.png"); fig("04_acf_pacf.png")""")
code("""pd.DataFrame(eda["top_anomalies"])""")
code("""fig("05_anomalies.png"); fig("06_calendar_heatmap.png"); fig("07_submetering_composition.png")""")

md("""## 3. Time-based features

Calendar features, cyclical encodings, holiday flags, Fourier harmonics for the yearly cycle,
and lag / rolling statistics of the target. The direct GBM uses lags shifted by the 28-day horizon.""")
code("""from src.features import calendar_features, fourier_terms, lag_features
cal = calendar_features(y.index); four = fourier_terms(y.index, K=3); lags = lag_features(y)
display(cal.tail(3)); display(four.tail(3).round(3)); display(lags.tail(3).round(2))""")
code("""meta = json.load(open(M / "run_meta.json"))
imp = pd.Series(meta["gbm_importance"]).sort_values()
imp.plot.barh(figsize=(7,4), title="GBM permutation importance (train+val)"); plt.show()""")

md("""## 4. Models and rolling-origin backtest

Protocol: tune on validation, refit on train+validation, then 28-day forecasts from 7 origins
spaced 14 days apart in the hold-out (196 points per model).""")
code("""val_m = pd.read_csv(M / "validation_metrics.csv", index_col=0)
test_m = pd.read_csv(M / "test_metrics.csv", index_col=0)
display(Markdown("**Validation window**")); display(val_m.round(3))
display(Markdown("**Hold-out window**")); display(test_m.round(3))""")
code("""fig("09_test_metrics.png"); fig("11_forecasts_vs_actual.png")""")
md("### SARIMA order search and Prophet grid")
code("""for f in ["sarima_order_search.csv", "prophet_grid.csv", "lstm_grid.csv", "gru_grid.csv"]:
    p = M / f
    if p.exists():
        display(Markdown(f"**{f}**")); display(pd.read_csv(p).head(8))""")
md("### Live re-fit: SARIMAX with Fourier regressors on the training window")
code("""from src.models.arima import SarimaFourier
s = meta["sarima"]
m = SarimaFourier(order=tuple(s["order"]), seasonal_order=tuple(s["seasonal_order"]), K=s["fourier_K"], search=False).fit(train)
print(m.res_.summary().tables[1])""")
md("### Live re-fit: Prophet components")
code("""from src.models.prophet_model import ProphetModel
p = meta["prophet"]
pm = ProphetModel(p["cps"], p["sps"], p["mode"]).fit(pd.concat([train, val]))
future = pm.model_.make_future_dataframe(periods=28)
fcst = pm.model_.predict(future)
pm.model_.plot_components(fcst); plt.show()""")
code("""fig("08_rnn_training_curves.png")""")

md("## 5. Error by horizon and business impact")
code("""byh = pd.read_csv(M / "test_mae_by_horizon.csv", index_col=0)
display(byh.loc[[1,7,14,21,28]].round(2)); fig("10_mae_by_horizon.png")""")
code("""bt = pd.read_csv(M / "business_impact.csv", index_col=0)
cols = ["mean_abs_error","share_days_under_forecast","annual_cost_portfolio","annual_buffer_cost_portfolio"]
display(bt[cols].round(2)); fig("13_business_impact.png")
display(Markdown("**Safety buffer calibrated on validation, service level achieved on hold-out**"))
display(pd.read_csv(M / "service_level.csv", index_col=0).round(3))""")

md("""## 6. Drift detection and retraining

PSI / KS on seasonally adjusted demand, Page-Hinkley and CUSUM on residuals, and a rolling-MAE
monitor, all simulated over the hold-out. The operating rules are in
`docs/retraining_and_drift_plan.md`.""")
code("""drift = json.load(open(M / "drift_report.json"))
dd = drift["distribution_drift_train_vs_test"]
display(pd.DataFrame({"raw": dd["raw"], "seasonally_adjusted": dd["seasonally_adjusted"]}).round(3))
print("verdict:", dd["verdict"]); print("Page-Hinkley alarms:", drift["page_hinkley_alarms"])
print("CUSUM alarms:", drift["cusum_alarms"]); display(pd.Series(drift["performance"]).round(3).to_frame("value"))
fig("12_drift_monitoring.png")""")
code("""# live: PSI of the latest 90 days vs the same window a year earlier, seasonally adjusted
from src.drift import psi, seasonal_adjust
adj = seasonal_adjust(y)
cur = adj.iloc[-90:]; prev = adj.loc[cur.index - pd.Timedelta(days=365)]
print(f"PSI last 90 days vs previous year: {psi(prev.values, cur.values):.3f}")""")
md("""### Retraining plan in one table

| cadence | action |
|---|---|
| daily | ingest + data checks, Prophet refit, SARIMAX filter update, residual monitors |
| weekly | SARIMAX + GBM refit, PSI/KS drift check, interval coverage |
| monthly | LSTM/GRU retrain, ensemble re-weighting, champion/challenger |
| quarterly | SARIMA order and GBM hyper-parameter re-selection, cost assumptions review |
| on alarm | triggered retrain with backtest gate and rollback |""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("demand_forecasting_walkthrough.ipynb")
nbf.write(nb, out)
print("wrote", out)
