"""Exploratory analysis: trend, seasonality, stationarity, anomalies.

Produces figures in reports/figures and a JSON summary in reports/metrics.
"""
from __future__ import annotations

import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import MSTL, STL
from statsmodels.tsa.stattools import adfuller, kpss

from . import config
from .data import load_daily, target_series, train_val_test_split

PALETTE = {
    "series": "#4C72B0", "trend": "#DD8452", "test": "#C44E52",
    "val": "#8172B3", "anomaly": "#C44E52", "imputed": "#937860",
}
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 130, "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
})


def _save(fig, name: str):
    out = config.FIGURES / name
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return str(out.relative_to(config.ROOT)).replace("\\", "/")


def stationarity_tests(y: pd.Series) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf = adfuller(y.dropna(), autolag="AIC")
        kp = kpss(y.dropna(), regression="c", nlags="auto")
        adf_d7 = adfuller(y.diff(7).dropna(), autolag="AIC")
    return {
        "adf_stat": float(adf[0]), "adf_pvalue": float(adf[1]),
        "kpss_stat": float(kp[0]), "kpss_pvalue": float(kp[1]),
        "adf_seasonal_diff7_pvalue": float(adf_d7[1]),
        "interpretation": _interpret(adf[1], kp[1]),
    }


def _interpret(adf_p: float, kpss_p: float) -> str:
    if adf_p < 0.05 and kpss_p >= 0.05:
        return ("ADF rejects a unit root and KPSS does not reject stationarity: the level "
                "series is stationary (d=0). Seasonality is handled with seasonal terms / "
                "Fourier regressors rather than regular differencing.")
    if adf_p >= 0.05 and kpss_p < 0.05:
        return "Both tests indicate a unit root: use first differencing (d=1)."
    if adf_p < 0.05 and kpss_p < 0.05:
        return ("ADF rejects a unit root but KPSS rejects level-stationarity: difference-"
                "stationary with slowly varying mean; try d=1 or strong seasonal terms.")
    return "Tests disagree / inconclusive: compare d=0 and d=1 by AIC."


def detect_anomalies(y: pd.Series, period: int = 7, z_thresh: float = 4.0) -> pd.DataFrame:
    """Robust anomaly flags on the STL remainder (MAD-based z-score)."""
    stl = STL(y, period=period, robust=True).fit()
    resid = stl.resid
    med = np.median(resid)
    mad = np.median(np.abs(resid - med)) * 1.4826
    z = (resid - med) / mad
    out = pd.DataFrame({"y": y, "trend": stl.trend, "seasonal": stl.seasonal,
                        "resid": resid, "robust_z": z})
    out["is_anomaly"] = out["robust_z"].abs() > z_thresh
    out["direction"] = np.where(out["robust_z"] > 0, "spike", "dip")
    return out


def run_eda(dataset: str | None = None) -> dict:
    dataset = dataset or config.DATASET
    unit = config.TARGET_UNIT[dataset]
    df = load_daily(dataset)
    y = target_series(dataset)
    train, val, test = train_val_test_split(y)
    figs = {}

    # 1. Full series + trend + split shading ----------------------------------
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(y.index, y, color=PALETTE["series"], lw=0.7, alpha=0.8, label="daily")
    ax.plot(y.index, y.rolling(28, center=True).mean(), color=PALETTE["trend"], lw=2,
            label="28-day centred mean (trend)")
    ax.axvspan(val.index[0], val.index[-1], color=PALETTE["val"], alpha=0.12, label="validation")
    ax.axvspan(test.index[0], test.index[-1], color=PALETTE["test"], alpha=0.12, label="test")
    imp = df.index[df["was_imputed"]]
    if len(imp):
        ax.scatter(imp, y.reindex(imp), color=PALETTE["imputed"], s=12, zorder=5,
                   label=f"imputed days (n={len(imp)})")
    ax.set_title(f"Daily {y.name} ({unit}) with trend and evaluation windows")
    ax.set_ylabel(unit)
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    figs["series"] = _save(fig, "01_series_trend.png")

    # 2. Seasonality: weekly + yearly -----------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    dow = y.groupby(y.index.dayofweek)
    axes[0].boxplot([g.values for _, g in dow], tick_labels=list("MTWTFSS"), showfliers=False)
    axes[0].set_title("Weekly profile (by day of week)")
    axes[0].set_ylabel(unit)
    mon = y.groupby(y.index.month)
    axes[1].boxplot([g.values for _, g in mon], tick_labels=list("JFMAMJJASOND"), showfliers=False)
    axes[1].set_title("Yearly profile (by month)")
    for yr, g in y.groupby(y.index.year):
        axes[2].plot(g.index.dayofyear, g.rolling(14, center=True).mean(), lw=1.2, label=str(yr))
    axes[2].set_title("Year-over-year overlay (14-day smoothed)")
    axes[2].set_xlabel("day of year")
    axes[2].legend(fontsize=8)
    figs["seasonality"] = _save(fig, "02_seasonality_profiles.png")

    # 3. Multi-seasonal decomposition -----------------------------------------
    mstl = MSTL(y, periods=(7, 365), stl_kwargs={"robust": True}).fit()
    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(y, lw=0.6); axes[0].set_title("observed")
    axes[1].plot(mstl.trend, color=PALETTE["trend"]); axes[1].set_title("trend")
    axes[2].plot(mstl.seasonal.iloc[:, 0], lw=0.5); axes[2].set_title("seasonal (weekly, period 7)")
    axes[3].plot(mstl.seasonal.iloc[:, 1]); axes[3].set_title("seasonal (yearly, period 365)")
    axes[4].plot(mstl.resid, lw=0.5, color="grey"); axes[4].set_title("remainder")
    figs["decomposition"] = _save(fig, "03_mstl_decomposition.png")
    var_total = float(np.var(y))
    strength = {
        "weekly": float(max(0, 1 - np.var(mstl.resid) / np.var(mstl.resid + mstl.seasonal.iloc[:, 0]))),
        "yearly": float(max(0, 1 - np.var(mstl.resid) / np.var(mstl.resid + mstl.seasonal.iloc[:, 1]))),
        "trend": float(max(0, 1 - np.var(mstl.resid) / np.var(mstl.resid + mstl.trend))),
    }

    # 4. ACF / PACF ------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    plot_acf(y, lags=60, ax=axes[0, 0], title="ACF - levels")
    plot_pacf(y, lags=60, ax=axes[0, 1], title="PACF - levels", method="ywm")
    plot_acf(y.diff(7).dropna(), lags=60, ax=axes[1, 0], title="ACF - seasonal diff (7)")
    plot_pacf(y.diff(7).dropna(), lags=60, ax=axes[1, 1], title="PACF - seasonal diff (7)", method="ywm")
    figs["acf"] = _save(fig, "04_acf_pacf.png")

    # 5. Anomalies ---------------------------------------------------------------
    an = detect_anomalies(y)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(y, lw=0.6, color=PALETTE["series"])
    flagged = an[an["is_anomaly"]]
    axes[0].scatter(flagged.index, flagged["y"], color=PALETTE["anomaly"], s=18, zorder=5,
                    label=f"anomalies |robust z|>4 (n={len(flagged)})")
    axes[0].legend(); axes[0].set_title("Anomalies on STL remainder")
    axes[1].plot(an["robust_z"], lw=0.5, color="grey")
    axes[1].axhline(4, color=PALETTE["anomaly"], ls="--"); axes[1].axhline(-4, color=PALETTE["anomaly"], ls="--")
    axes[1].set_title("robust z-score of remainder")
    figs["anomalies"] = _save(fig, "05_anomalies.png")
    flagged.reset_index().rename(columns={"index": "date"}).to_csv(
        config.METRICS / "anomalies.csv", index=False)

    # 6. Calendar heat-map --------------------------------------------------------
    piv = pd.DataFrame({"y": y, "year": y.index.year, "doy": y.index.dayofyear}).pivot_table(
        index="year", columns="doy", values="y")
    fig, ax = plt.subplots(figsize=(15, 3.2))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set_xlabel("day of year"); ax.set_title(f"Calendar heat-map of daily {unit}")
    ax.grid(False)
    fig.colorbar(im, ax=ax, pad=0.01)
    figs["heatmap"] = _save(fig, "06_calendar_heatmap.png")

    # 7. Dataset-specific: sub-metering composition ---------------------------------
    sub_cols = [c for c in df.columns if c.startswith("sub_metering")]
    if sub_cols:
        monthly = (df[sub_cols] / 1000).resample("ME").sum()
        monthly["other"] = y.resample("ME").sum() - monthly.sum(axis=1)
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.stackplot(monthly.index, monthly.T.values,
                     labels=["kitchen", "laundry", "heater + AC", "other (lighting etc.)"], alpha=0.85)
        ax.set_title("Monthly energy by circuit (kWh)"); ax.legend(loc="upper right", ncol=4, fontsize=8)
        figs["submetering"] = _save(fig, "07_submetering_composition.png")

    # Summary ------------------------------------------------------------------------
    missing = df["was_imputed"].sum()
    summary = {
        "dataset": dataset,
        "n_days": int(len(y)),
        "start": str(y.index[0].date()), "end": str(y.index[-1].date()),
        "mean": float(y.mean()), "std": float(y.std()),
        "min": float(y.min()), "max": float(y.max()),
        "imputed_days": int(missing),
        "imputed_share_%": float(100 * missing / len(y)),
        "weekday_mean": float(y[y.index.dayofweek < 5].mean()),
        "weekend_mean": float(y[y.index.dayofweek >= 5].mean()),
        "winter_mean(DJF)": float(y[y.index.month.isin([12, 1, 2])].mean()),
        "summer_mean(JJA)": float(y[y.index.month.isin([6, 7, 8])].mean()),
        "yoy_trend_%_per_year": float(
            100 * np.polyfit(np.arange(len(y)) / 365.25, y.values, 1)[0] / y.mean()),
        "seasonal_strength": strength,
        "stationarity": stationarity_tests(y),
        "n_anomalies": int(an["is_anomaly"].sum()),
        "anomaly_z_threshold": 4.0,
        "top_anomalies": [
            {"date": str(d.date()), "value": round(float(r.y), 2), "robust_z": round(float(r.robust_z), 2),
             "type": r.direction}
            for d, r in flagged.reindex(flagged["robust_z"].abs().sort_values(ascending=False).index).head(10).iterrows()
        ],
        "split": {"train": [str(train.index[0].date()), str(train.index[-1].date()), len(train)],
                  "val": [str(val.index[0].date()), str(val.index[-1].date()), len(val)],
                  "test": [str(test.index[0].date()), str(test.index[-1].date()), len(test)]},
        "figures": figs,
    }
    with open(config.METRICS / "eda_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


if __name__ == "__main__":
    s = run_eda()
    print(json.dumps({k: v for k, v in s.items() if k not in ("figures", "top_anomalies")}, indent=2))
    print(pd.DataFrame(s["top_anomalies"]))
