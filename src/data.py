"""Data acquisition and preparation.

Two sources are supported:

* ``uci_power``: UCI *Individual Household Electric Power Consumption*
  (2,075,259 one-minute readings, Dec-2006 .. Nov-2010). Aggregated to a daily
  energy series (kWh) with explicit handling of missing minutes.
* ``m5``: Kaggle *M5 Forecasting - Accuracy*. Aggregated to total daily unit
  sales. Only used if the Kaggle CSVs are present (they require a Kaggle login).
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from . import config

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
UCI_DIR = config.DATA_RAW / "uci_power"
UCI_FILE = UCI_DIR / "household_power_consumption.txt"
M5_DIR = config.DATA_RAW / "m5"

MIN_DAILY_COVERAGE = 0.90  # fraction of the 1440 minutes that must be observed


def download_uci_power(force: bool = False) -> Path:
    """Download and unzip the UCI dataset if it is not already present."""
    UCI_DIR.mkdir(parents=True, exist_ok=True)
    if UCI_FILE.exists() and not force:
        return UCI_FILE
    print(f"Downloading {UCI_URL} ...")
    with urlopen(UCI_URL, timeout=120) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(UCI_DIR)
    return UCI_FILE


def load_uci_power_daily() -> pd.DataFrame:
    """Return a daily DataFrame indexed by date with the target and QA columns.

    Columns
    -------
    daily_kwh      : energy consumed that day (kWh), interpolated where coverage is low
    coverage       : fraction of minutes with a valid reading
    was_imputed    : True if the day was reconstructed by interpolation
    sub_metering_* : daily Wh of kitchen / laundry / heater+AC circuits
    """
    cached = config.DATA_PROCESSED / "uci_power_daily.parquet"
    if cached.exists():
        return pd.read_parquet(cached)

    path = download_uci_power()
    raw = pd.read_csv(
        path,
        sep=";",
        na_values=["?"],
        usecols=["Date", "Time", "Global_active_power",
                 "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"],
        dtype={"Global_active_power": "float64", "Sub_metering_1": "float64",
               "Sub_metering_2": "float64", "Sub_metering_3": "float64"},
        low_memory=False,
    )
    raw["date"] = pd.to_datetime(raw["Date"], format="%d/%m/%Y")
    # Global_active_power is in kW averaged over one minute -> kWh = kW / 60
    raw["kwh"] = raw["Global_active_power"] / 60.0

    daily = raw.groupby("date").agg(
        daily_kwh=("kwh", "sum"),
        n_valid=("kwh", "count"),
        sub_metering_1=("Sub_metering_1", "sum"),
        sub_metering_2=("Sub_metering_2", "sum"),
        sub_metering_3=("Sub_metering_3", "sum"),
    )
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx)
    daily.index.name = "date"
    daily["coverage"] = daily["n_valid"].fillna(0) / 1440.0

    # Scale partially observed days up to a full day, then drop days with poor
    # coverage so a half-observed day is not mistaken for a demand collapse.
    partial = (daily["coverage"] > 0) & (daily["coverage"] < 1)
    daily.loc[partial, "daily_kwh"] = (
        daily.loc[partial, "daily_kwh"] / daily.loc[partial, "coverage"]
    )
    low = daily["coverage"] < MIN_DAILY_COVERAGE
    daily.loc[low, "daily_kwh"] = np.nan
    daily["was_imputed"] = daily["daily_kwh"].isna()
    daily["daily_kwh"] = daily["daily_kwh"].interpolate(
        method="time", limit_direction="both"
    )
    # first and last days are partial (recording starts 17:24 and ends 21:02)
    daily = daily.iloc[1:-1]
    daily = daily.drop(columns=["n_valid"])
    daily.to_parquet(cached)
    return daily


def load_m5_daily() -> pd.DataFrame:
    """Aggregate M5 to a total daily unit-sales series (all stores, all items)."""
    cached = config.DATA_PROCESSED / "m5_daily.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    sales_file = M5_DIR / "sales_train_evaluation.csv"
    if not sales_file.exists():
        sales_file = M5_DIR / "sales_train_validation.csv"
    cal_file = M5_DIR / "calendar.csv"
    if not (sales_file.exists() and cal_file.exists()):
        raise FileNotFoundError(
            "M5 files not found. Download from "
            "https://www.kaggle.com/competitions/m5-forecasting-accuracy/data "
            f"and place sales_train_evaluation.csv + calendar.csv in {M5_DIR}"
        )
    sales = pd.read_csv(sales_file)
    cal = pd.read_csv(cal_file, parse_dates=["date"])
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    totals = sales[day_cols].sum(axis=0)
    totals.index.name = "d"
    df = totals.rename("units_sold").reset_index().merge(cal[["d", "date"]], on="d")
    df = df.set_index("date").sort_index()[["units_sold"]]
    df.index.name = "date"
    df["was_imputed"] = False
    df["coverage"] = 1.0
    df.to_parquet(cached)
    return df


def load_daily(dataset: str | None = None) -> pd.DataFrame:
    dataset = dataset or config.DATASET
    if dataset == "uci_power":
        return load_uci_power_daily()
    if dataset == "m5":
        return load_m5_daily()
    raise ValueError(f"Unknown dataset {dataset!r}")


def target_series(dataset: str | None = None) -> pd.Series:
    dataset = dataset or config.DATASET
    df = load_daily(dataset)
    s = df[config.TARGET_NAME[dataset]].astype(float).copy()
    s.name = config.TARGET_NAME[dataset]
    s.index = pd.DatetimeIndex(s.index, freq="D")
    return s


def train_val_test_split(y: pd.Series):
    """Chronological split: [train | VAL_DAYS validation | TEST_DAYS test]."""
    test = y.iloc[-config.TEST_DAYS:]
    val = y.iloc[-(config.TEST_DAYS + config.VAL_DAYS):-config.TEST_DAYS]
    train = y.iloc[: -(config.TEST_DAYS + config.VAL_DAYS)]
    return train, val, test


if __name__ == "__main__":
    s = target_series()
    print(s.describe())
    print(s.head(), s.tail())
