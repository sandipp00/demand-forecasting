"""Data-drift and model-drift detection utilities.

Three complementary monitors are implemented and exercised on the dataset:

1. **Input / target distribution drift** between a reference window (training
   data) and a live window: Population Stability Index (PSI) and the two-sample
   Kolmogorov-Smirnov test, computed on the raw target *and* on the seasonally
   adjusted target (so that expected seasonal change does not raise alarms).
2. **Sequential change detection** on forecast residuals with the
   Page-Hinkley test (detects a persistent shift of the mean error = bias).
3. **Performance drift**: rolling MAE of the deployed model compared with the
   MAE established during validation; alarm when it exceeds a tolerance band.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import config


# --------------------------------------------------------------------------- #
# Distribution drift
# --------------------------------------------------------------------------- #
def psi(reference: np.ndarray, current: np.ndarray, bins: int | None = None, eps: float = 1e-6) -> float:
    """Population Stability Index. Rule of thumb: <0.1 stable, 0.1-0.25 moderate, >0.25 severe.

    Bins default to min(10, n/30) so small live windows do not inflate the index."""
    if bins is None:
        bins = int(max(4, min(10, min(len(reference), len(current)) // 30)))
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_p = np.histogram(reference, bins=edges)[0] / len(reference) + eps
    cur_p = np.histogram(current, bins=edges)[0] / len(current) + eps
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def ks_test(reference: np.ndarray, current: np.ndarray) -> dict:
    stat, p = stats.ks_2samp(reference, current)
    return {"ks_stat": float(stat), "ks_pvalue": float(p)}


def seasonal_adjust(y: pd.Series) -> pd.Series:
    """Remove a smoothed day-of-year profile and the day-of-week profile.

    The day-of-year means are smoothed with a circular 31-day window because
    only a few years are available for each calendar day."""
    doy_mean = y.groupby(y.index.dayofyear).mean().reindex(range(1, 367)).interpolate(limit_direction="both")
    padded = pd.concat([doy_mean.iloc[-15:], doy_mean, doy_mean.iloc[:15]])
    smooth = padded.rolling(31, center=True).mean().iloc[15:-15]
    smooth.index = doy_mean.index
    doy_prof = pd.Series(smooth.reindex(y.index.dayofyear).values, index=y.index)
    dow_prof = (y - doy_prof).groupby(y.index.dayofweek).transform("mean")
    return y - doy_prof - dow_prof + y.mean()


def distribution_drift(y: pd.Series, reference: pd.Series, current: pd.Series) -> dict:
    """PSI/KS on raw and seasonally adjusted target, plus moment comparison."""
    adj = seasonal_adjust(y)
    out = {
        "reference_window": [str(reference.index[0].date()), str(reference.index[-1].date())],
        "current_window": [str(current.index[0].date()), str(current.index[-1].date())],
        "raw": {"psi": psi(reference.values, current.values), **ks_test(reference.values, current.values),
                "ref_mean": float(reference.mean()), "cur_mean": float(current.mean()),
                "ref_std": float(reference.std()), "cur_std": float(current.std())},
        "seasonally_adjusted": {
            "psi": psi(adj.loc[reference.index].values, adj.loc[current.index].values),
            **ks_test(adj.loc[reference.index].values, adj.loc[current.index].values),
            "ref_mean": float(adj.loc[reference.index].mean()),
            "cur_mean": float(adj.loc[current.index].mean()),
        },
    }
    out["verdict"] = _psi_verdict(out["seasonally_adjusted"]["psi"])
    return out


def _psi_verdict(v: float) -> str:
    return "stable" if v < 0.10 else ("moderate drift - investigate" if v < 0.25 else "severe drift - retrain")


# --------------------------------------------------------------------------- #
# Sequential change detection on residuals
# --------------------------------------------------------------------------- #
def page_hinkley(x: np.ndarray, delta: float = 0.005, threshold: float = 50.0, alpha: float = 0.999):
    """Page-Hinkley test for an increase in the mean of `x` (two-sided via +/- x).

    Returns indices where an alarm fires (the detector is reset after each alarm).
    """
    alarms = []
    for sign in (+1, -1):
        mean, cum, cum_min = 0.0, 0.0, 0.0
        n = 0
        for i, v in enumerate(sign * x):
            n += 1
            mean = mean + (v - mean) / n
            cum = alpha * cum + (v - mean - delta)
            cum_min = min(cum_min, cum)
            if cum - cum_min > threshold:
                alarms.append((i, "up" if sign > 0 else "down"))
                mean, cum, cum_min, n = 0.0, 0.0, 0.0, 0
    return sorted(alarms)


def cusum(x: np.ndarray, k: float, h: float):
    """Tabular CUSUM for mean shift (k = allowance, h = decision interval). Returns alarm idx."""
    s_pos = s_neg = 0.0
    alarms = []
    for i, v in enumerate(x):
        s_pos = max(0.0, s_pos + v - k)
        s_neg = max(0.0, s_neg - v - k)
        if s_pos > h or s_neg > h:
            alarms.append(i)
            s_pos = s_neg = 0.0
    return alarms


# --------------------------------------------------------------------------- #
# Performance drift
# --------------------------------------------------------------------------- #
def performance_monitor(residuals: pd.Series, baseline_mae: float, window: int = 28,
                        tolerance: float = 1.25) -> pd.DataFrame:
    """Rolling MAE & bias vs. the validation MAE; flags windows above tolerance."""
    df = pd.DataFrame({"resid": residuals})
    df["rolling_mae"] = df["resid"].abs().rolling(window).mean()
    df["rolling_bias"] = df["resid"].rolling(window).mean()
    df["threshold"] = baseline_mae * tolerance
    df["alarm"] = df["rolling_mae"] > df["threshold"]
    return df


# --------------------------------------------------------------------------- #
# End-to-end drift report
# --------------------------------------------------------------------------- #
def run_drift_analysis(y: pd.Series, train: pd.Series, val: pd.Series, test: pd.Series,
                       fc_test_best: pd.DataFrame, val_mae: float, model_name: str) -> dict:
    """Simulate the monitoring stack over the hold-out period and save a figure."""
    # 1) distribution drift: last year of training vs test window
    ref = train.iloc[-365:]
    dd_test = distribution_drift(y, ref, test)
    # year-over-year PSI timeline (each quarter vs same quarter one year earlier)
    adj = seasonal_adjust(y)
    timeline = []
    q_ends = pd.date_range(y.index[0] + pd.Timedelta(days=365 + 90), y.index[-1], freq="QE")
    for end in q_ends:
        cur = adj.loc[end - pd.Timedelta(days=89): end]
        prev = adj.loc[end - pd.Timedelta(days=365 + 89): end - pd.Timedelta(days=365)]
        if len(cur) > 60 and len(prev) > 60:
            timeline.append({"quarter_end": end, "psi_vs_prev_year": psi(prev.values, cur.values),
                             "mean_shift": float(cur.mean() - prev.mean())})
    timeline = pd.DataFrame(timeline)

    # 2) residual monitoring on the deployed (best) model: keep horizons up to the
    #    re-forecast cadence (ORIGIN_STEP) so every day has exactly one "live" residual
    short = fc_test_best[fc_test_best["h"] <= config.ORIGIN_STEP]
    resid = (short.groupby("date").apply(lambda g: (g["yhat"] - g["y"]).mean(), include_groups=False)).sort_index()
    perf = performance_monitor(resid, baseline_mae=val_mae, window=28, tolerance=1.25)
    ph = page_hinkley(resid.values, delta=0.05 * val_mae, threshold=8 * val_mae)
    cs = cusum(resid.values / resid.std(), k=0.5, h=5.0)

    # 3) figure
    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    axes[0].bar(timeline["quarter_end"], timeline["psi_vs_prev_year"], width=60, color="#4C72B0")
    axes[0].axhline(0.10, color="orange", ls="--", label="moderate (0.10)")
    axes[0].axhline(0.25, color="red", ls="--", label="severe (0.25)")
    axes[0].set_title("Data drift: PSI of seasonally-adjusted demand, each quarter vs. same quarter previous year")
    axes[0].legend(fontsize=8)
    axes[1].plot(resid.index, resid.values, lw=0.8, color="grey", label=f"residual (yhat - y), h<={config.ORIGIN_STEP}")
    axes[1].plot(perf.index, perf["rolling_bias"], color="#DD8452", lw=2, label="28-day rolling bias")
    for i, d in ph:
        axes[1].axvline(resid.index[i], color="red", ls=":", lw=1.5)
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_title(f"Model drift: {model_name} residuals on hold-out; red dotted = Page-Hinkley alarm ({len(ph)})")
    axes[1].legend(fontsize=8)
    axes[2].plot(perf.index, perf["rolling_mae"], color="#4C72B0", lw=2, label="28-day rolling MAE")
    axes[2].plot(perf.index, perf["threshold"], color="red", ls="--", label=f"alarm threshold = 1.25 x validation MAE ({val_mae:.2f})")
    axes[2].axhline(val_mae, color="green", ls=":", label="validation MAE")
    axes[2].fill_between(perf.index, 0, perf["rolling_mae"].max() * 1.1, where=perf["alarm"].fillna(False),
                         color="red", alpha=0.12, label="alarm window")
    axes[2].set_ylim(0, perf["rolling_mae"].max() * 1.1)
    axes[2].set_title("Performance drift: rolling MAE vs. tolerance band")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig_path = config.FIGURES / "12_drift_monitoring.png"
    fig.savefig(fig_path)
    plt.close(fig)

    report = {
        "monitored_model": model_name,
        "validation_mae_baseline": float(val_mae),
        "distribution_drift_train_vs_test": dd_test,
        "psi_timeline": [{"quarter_end": str(r.quarter_end.date()), "psi": round(r.psi_vs_prev_year, 4),
                          "mean_shift": round(r.mean_shift, 3)} for r in timeline.itertuples()],
        "page_hinkley_alarms": [{"date": str(resid.index[i].date()), "direction": d} for i, d in ph],
        "cusum_alarms": [str(resid.index[i].date()) for i in cs],
        "performance": {
            "test_mae_live": float(resid.abs().mean()),
            "test_bias_live": float(resid.mean()),
            "days_in_alarm": int(perf["alarm"].sum()),
            "share_days_in_alarm": float(perf["alarm"].mean()),
            "max_rolling_mae": float(perf["rolling_mae"].max()),
        },
        "figure": str(fig_path.relative_to(config.ROOT)).replace("\\", "/"),
    }
    with open(config.METRICS / "drift_report.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    perf.to_csv(config.METRICS / "drift_performance_monitor.csv")
    return report
