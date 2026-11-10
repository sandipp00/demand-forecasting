"""End-to-end pipeline: data -> EDA -> tuning -> backtests -> business impact -> drift -> report.

Usage
-----
    python run_pipeline.py                 # full run (~10-15 min on CPU)
    python run_pipeline.py --fast          # skip grid searches, fewer epochs (smoke test)
    python run_pipeline.py --skip-eda      # reuse existing EDA outputs
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.backtest import add_ensemble, rolling_forecasts, summarise
from src.data import target_series, train_val_test_split
from src.drift import run_drift_analysis
from src.eda import run_eda
from src.evaluate import business_table, metrics_by_horizon
from src.models.arima import SarimaFourier
from src.models.baselines import LastYearNaive, SeasonalMean, SeasonalNaive
from src.models.deep import RNNForecaster
from src.models.gbm import GBMDirect
from src.models.prophet_model import ProphetModel
from src.report import write_report

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(config.FIGURES / name)
    plt.close(fig)


# ----------------------------------------------------------------------------- tuning
def tune_sarima(train, fast):
    log("SARIMA: order search by AIC (weekly seasonal ARIMA + yearly Fourier K=3)")
    if fast:
        m = SarimaFourier(order=(1, 0, 1), seasonal_order=(1, 1, 1, 7), search=False).fit(train)
    else:
        m = SarimaFourier(K=3).fit(train)
        m.search_log_.to_csv(config.METRICS / "sarima_order_search.csv", index=False)
    log(f"  selected {m.name}  AIC={m.res_.aic:.1f}")
    return m


def tune_prophet(y, train, val, fast):
    log("Prophet: grid over changepoint / seasonality priors, scored by MAE on validation")
    grid = [(0.05, 10.0, "additive")] if fast else [
        (cps, sps, mode) for cps in (0.01, 0.05, 0.2) for sps in (1.0, 10.0)
        for mode in ("additive", "multiplicative")]
    rows = []
    for cps, sps, mode in grid:
        m = ProphetModel(cps, sps, mode, refit_each_origin=False).fit(train)
        fc = rolling_forecasts([m], y, val, verbose=False)
        mae = (fc["y"] - fc["yhat"]).abs().mean()
        rows.append({"cps": cps, "sps": sps, "mode": mode, "val_MAE": mae})
        log(f"  cps={cps:<5} sps={sps:<5} {mode:<14} val MAE={mae:.3f}")
    res = pd.DataFrame(rows).sort_values("val_MAE")
    res.to_csv(config.METRICS / "prophet_grid.csv", index=False)
    best = res.iloc[0]
    return ProphetModel(best.cps, best.sps, best["mode"], refit_each_origin=True).fit(train), res


def tune_rnn(cell, train, val, fast):
    log(f"{cell.upper()}: training with early stopping on validation")
    grid = [dict(hidden=32, layers=1)] if fast else [dict(hidden=32, layers=1), dict(hidden=64, layers=1),
                                                      dict(hidden=64, layers=2)]
    best = None
    rows = []
    for g in grid:
        m = RNNForecaster(cell, epochs=15 if fast else 200, **g).fit(train, val)
        rows.append({**g, "best_val_mse_scaled": m.best_val_loss_, "epochs": m.epochs_run_})
        log(f"  {g}  val MSE(scaled)={m.best_val_loss_:.4f}  epochs={m.epochs_run_}")
        if best is None or m.best_val_loss_ < best.best_val_loss_:
            best = m
    pd.DataFrame(rows).to_csv(config.METRICS / f"{cell}_grid.csv", index=False)
    return best


# ----------------------------------------------------------------------------- plots
def plot_forecasts(fc, y, test, names, fname, title):
    fig, axes = plt.subplots(len(names), 1, figsize=(14, 3.2 * len(names)), sharex=True)
    axes = np.atleast_1d(axes)
    hist = y.loc[test.index[0] - pd.Timedelta(days=56): test.index[-1]]
    for ax, name in zip(axes, names):
        g = fc[fc["model"] == name]
        ax.plot(hist.index, hist.values, color="black", lw=0.9, label="actual")
        for k, (origin, go) in enumerate(g.groupby("origin")):
            ax.plot(go["date"], go["yhat"], lw=1.3, alpha=0.9,
                    color=plt.cm.viridis(k / max(1, g["origin"].nunique() - 1)),
                    label="28-day forecasts from each origin" if k == 0 else None)
            if go["lo"].notna().any() and k % 2 == 0:
                ax.fill_between(go["date"], go["lo"], go["hi"], alpha=0.12, color="grey")
        mae = (g["y"] - g["yhat"]).abs().mean()
        ax.set_title(f"{name}   (test MAE {mae:.2f})", fontsize=10)
        ax.axvline(test.index[0], color="red", ls="--", lw=0.8)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(title, y=1.0)
    save_fig(fig, fname)


def plot_metric_bars(summary, fname, unit):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    s = summary.sort_values("MAE")
    colors = ["#C44E52" if "Naive" in n or "Mean(" in n or "LastYear" in n else "#4C72B0" for n in s.index]
    for ax, col, lab in zip(axes, ["MAE", "MAPE_%", "MASE"], [f"MAE ({unit})", "MAPE (%)", "MASE (vs seasonal naive)"]):
        ax.barh(s.index, s[col], color=colors)
        ax.invert_yaxis(); ax.set_title(lab)
        for i, v in enumerate(s[col]):
            ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)
    axes[2].axvline(1.0, color="black", ls="--", lw=0.8)
    fig.suptitle("Hold-out accuracy by model (red = naive baselines)")
    save_fig(fig, fname)


def plot_horizon(byh, fname, unit):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for col in byh.columns:
        ax.plot(byh.index, byh[col], marker="o", ms=3, lw=1.4, label=col)
    ax.set_xlabel("forecast horizon (days ahead)"); ax.set_ylabel(f"MAE ({unit})")
    ax.set_title("Error growth with horizon (hold-out, all origins)")
    ax.legend(fontsize=8, ncol=2)
    save_fig(fig, fname)


def plot_business(bt, fname):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    cur = bt["currency"].iloc[0]
    b = bt.sort_values("annual_cost_portfolio")
    axes[0].barh(b.index, b["annual_cost_portfolio"] / 1e6, color="#4C72B0")
    axes[0].invert_yaxis(); axes[0].set_title(f"Annual cost of forecast error, portfolio ({cur} millions)")
    for i, v in enumerate(b["annual_cost_portfolio"] / 1e6):
        axes[0].text(v, i, f" {v:.2f}M", va="center", fontsize=8)
    axes[1].barh(b.index, b["annual_buffer_cost_portfolio"] / 1e6, color="#DD8452")
    axes[1].invert_yaxis(); axes[1].set_title(f"Annual cost of safety buffer for 95% service ({cur} millions)")
    for i, v in enumerate(b["annual_buffer_cost_portfolio"] / 1e6):
        axes[1].text(v, i, f" {v:.2f}M", va="center", fontsize=8)
    save_fig(fig, fname)


def plot_training_curves(rnns, fname):
    fig, axes = plt.subplots(1, len(rnns), figsize=(6 * len(rnns), 4))
    for ax, m in zip(np.atleast_1d(axes), rnns):
        h = pd.DataFrame(m.history_)
        ax.plot(h["train_loss"], label="train"); ax.plot(h["val_loss"], label="validation")
        ax.set_title(f"{m.name} training (hidden={m.hidden}, layers={m.layers}), early stop @ {len(h)}")
        ax.set_xlabel("epoch"); ax.set_ylabel("MSE (standardised)"); ax.legend()
    save_fig(fig, fname)


# ----------------------------------------------------------------------------- main
def main(fast=False, skip_eda=False):
    t_start = time.time()
    dataset = config.DATASET
    unit = config.TARGET_UNIT[dataset]
    y = target_series(dataset)
    train, val, test = train_val_test_split(y)
    log(f"dataset={dataset}  n={len(y)}  train={len(train)} val={len(val)} test={len(test)}")

    # 1. EDA -----------------------------------------------------------------
    if skip_eda and (config.METRICS / "eda_summary.json").exists():
        eda = json.load(open(config.METRICS / "eda_summary.json"))
    else:
        log("EDA: trend / seasonality / anomalies")
        eda = run_eda(dataset)

    # 2. Tune on validation -------------------------------------------------------
    baselines = [SeasonalNaive(), SeasonalMean(), LastYearNaive()]
    sarima = tune_sarima(train, fast)
    sarima_plain = SarimaFourier(order=sarima.order, seasonal_order=sarima.seasonal_order,
                                 use_fourier=False, search=False).fit(train)
    prophet, prophet_grid = tune_prophet(y, train, val, fast)
    gbm = GBMDirect().fit(train)
    lstm = tune_rnn("lstm", train, val, fast)
    gru = tune_rnn("gru", train, val, fast)
    plot_training_curves([lstm, gru], "08_rnn_training_curves.png")

    # 3. Validation backtest (models fit on train only) --------------------------------
    log("Validation backtest (rolling origins, models trained on train only)")
    val_models = baselines + [sarima_plain, sarima, prophet, gbm, lstm, gru]
    prophet.refit_each_origin = True
    fc_val = rolling_forecasts(val_models, y, val)
    val_summary = summarise(fc_val, train)
    # choose ensemble members = top-3 non-baseline models on validation
    non_base = [n for n in val_summary.index if not any(k in n for k in ("Naive", "Mean(", "LastYear"))]
    members = non_base[:3]
    fc_val = add_ensemble(fc_val, members)
    val_summary = summarise(fc_val, train)
    val_summary.to_csv(config.METRICS / "validation_metrics.csv")
    fc_val.to_csv(config.METRICS / "forecasts_validation.csv", index=False)
    log("validation MAE:\n" + val_summary["MAE"].round(3).to_string())

    # 4. Refit on train+val and run the hold-out test ------------------------------------
    log("Refitting on train+val for the hold-out test")
    trainval = pd.concat([train, val])
    sarima_final = SarimaFourier(order=sarima.order, seasonal_order=sarima.seasonal_order,
                                 K=sarima.K, search=False).fit(trainval)
    sarima_plain_final = SarimaFourier(order=sarima.order, seasonal_order=sarima.seasonal_order,
                                       use_fourier=False, search=False).fit(trainval)
    prophet_final = ProphetModel(prophet.cps, prophet.sps, prophet.mode, refit_each_origin=True).fit(trainval)
    gbm_final = GBMDirect().fit(trainval)
    rnn_final = []
    for m in (lstm, gru):
        best_epoch = max(5, int(np.argmin([h["val_loss"] for h in m.history_])) + 1)
        mf = RNNForecaster(m.cell, hidden=m.hidden, layers=m.layers, epochs=best_epoch, patience=10**6)
        mf.fit(trainval)  # fixed epoch budget learned on validation
        mf.name = m.name
        rnn_final.append(mf)
    test_models = baselines + [sarima_plain_final, sarima_final, prophet_final, gbm_final] + rnn_final
    log("Hold-out backtest (rolling origins)")
    fc_test = rolling_forecasts(test_models, y, test, with_intervals=True)
    timings = pd.Series(fc_test.attrs["timings"], name="forecast_seconds_7_origins")
    fc_test = add_ensemble(fc_test, members)
    test_summary = summarise(fc_test, trainval)
    test_summary.to_csv(config.METRICS / "test_metrics.csv")
    fc_test.to_csv(config.METRICS / "forecasts_test.csv", index=False)
    log("hold-out MAE:\n" + test_summary["MAE"].round(3).to_string())
    timings.to_csv(config.METRICS / "timings.csv")

    # per-horizon and plots
    byh = metrics_by_horizon(fc_test)
    byh.to_csv(config.METRICS / "test_mae_by_horizon.csv")
    plot_metric_bars(test_summary, "09_test_metrics.png", unit)
    plot_horizon(byh[[c for c in byh.columns if "Naive" not in c and "LastYear" not in c]],
                 "10_mae_by_horizon.png", unit)
    best = test_summary.index[0]
    show = [best] + [n for n in (sarima_final.name, prophet_final.name, "LSTM", "GRU", gbm_final.name)
                     if n != best][:4]
    plot_forecasts(fc_test, y, test, show, "11_forecasts_vs_actual.png",
                   "Hold-out forecasts (each colour = one 28-day forecast from a new origin; grey = 90% interval)")

    # 5. Business impact ------------------------------------------------------------------
    log("Business impact")
    bt = business_table(fc_test, dataset)
    bt.to_csv(config.METRICS / "business_impact.csv")
    plot_business(bt, "13_business_impact.png")
    # service-level check: buffer calibrated on validation residuals, applied on test
    b = config.BUSINESS[dataset]
    sl_rows = []
    for name, g in fc_test.groupby("model"):
        gv = fc_val[fc_val["model"] == name]
        if gv.empty:
            continue
        under_val = np.clip(gv["y"] - gv["yhat"], 0, None)
        buf = float(np.quantile(under_val, b["service_level"]))
        covered = float(((g["yhat"] + buf) >= g["y"]).mean())
        sl_rows.append({"model": name, "buffer_from_validation": buf, "achieved_service_level_test": covered,
                        "buffer_share_of_mean_demand_%": 100 * buf / test.mean()})
    sl = pd.DataFrame(sl_rows).set_index("model").sort_values("buffer_from_validation")
    sl.to_csv(config.METRICS / "service_level.csv")

    # 6. Drift ------------------------------------------------------------------------------
    log("Drift analysis")
    val_mae_best = float(val_summary.loc[best, "MAE"]) if best in val_summary.index else float(val_summary["MAE"].min())
    drift = run_drift_analysis(y, train, val, test, fc_test[fc_test["model"] == best], val_mae_best, best)

    # 7. Report -------------------------------------------------------------------------------
    meta = {
        "dataset": dataset, "unit": unit, "fast": fast,
        "sarima": sarima.summary_dict(), "sarima_name": sarima_final.name,
        "prophet": {"cps": prophet.cps, "sps": prophet.sps, "mode": prophet.mode, "name": prophet_final.name},
        "lstm": {"hidden": lstm.hidden, "layers": lstm.layers, "epochs": lstm.epochs_run_},
        "gru": {"hidden": gru.hidden, "layers": gru.layers, "epochs": gru.epochs_run_},
        "ensemble_members": members, "best_model": best,
        "runtime_minutes": (time.time() - t_start) / 60,
        "gbm_importance": gbm_final.feature_importance(trainval).head(12).round(4).to_dict(),
    }
    json.dump(meta, open(config.METRICS / "run_meta.json", "w"), indent=2, default=str)
    write_report(eda, val_summary, test_summary, byh, bt, sl, drift, meta, timings)
    log(f"done in {meta['runtime_minutes']:.1f} min -> reports/REPORT.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--skip-eda", action="store_true")
    a = ap.parse_args()
    main(fast=a.fast, skip_eda=a.skip_eda)
