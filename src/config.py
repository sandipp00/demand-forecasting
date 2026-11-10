"""Central configuration: paths, split sizes, horizon, business-cost assumptions."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
METRICS = REPORTS / "metrics"
MODELS_DIR = ROOT / "models_store"

for _p in (DATA_PROCESSED, FIGURES, METRICS, MODELS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# --- Dataset -----------------------------------------------------------------
# "uci_power": UCI Individual Household Electric Power Consumption (daily kWh).
# "m5":        Kaggle M5 Forecasting - Accuracy, aggregated to total daily unit
#              sales (requires sales_train_evaluation.csv + calendar.csv in data/raw/m5).
DATASET = "uci_power"
TARGET_NAME = {"uci_power": "daily_kwh", "m5": "units_sold"}
TARGET_UNIT = {"uci_power": "kWh", "m5": "units"}

# --- Forecast design -----------------------------------------------------------
HORIZON = 28            # days ahead per forecast origin (4 weeks)
TEST_DAYS = 112         # final hold-out window (4 x 28)
VAL_DAYS = 112          # validation window just before test, used for tuning
ORIGIN_STEP = 14        # days between rolling forecast origins inside the test window
SEASONAL_PERIOD = 7     # weekly seasonality for SARIMA
YEARLY_PERIOD = 365.25
LOOKBACK = 56           # input window for LSTM / GRU
RANDOM_SEED = 42

# --- Business-impact assumptions (editable) -------------------------------------
# Electricity retailer procuring energy day-ahead for a portfolio of households.
# Over-forecast  -> energy bought but not consumed, resold at a loss (EUR/kWh).
# Under-forecast -> shortfall bought on the intraday/balancing market at a premium.
BUSINESS = {
    "uci_power": {
        "portfolio_size": 100_000,          # households represented by this profile
        "cost_over_per_unit": 0.04,         # EUR per kWh over-procured
        "cost_under_per_unit": 0.12,        # EUR per kWh under-procured (balancing premium)
        "service_level": 0.95,              # target probability of covering demand
        "currency": "EUR",
    },
    "m5": {
        "portfolio_size": 1,
        "cost_over_per_unit": 1.50,         # holding + markdown cost per excess unit
        "cost_under_per_unit": 4.00,        # lost margin + goodwill per stock-out unit
        "service_level": 0.95,
        "currency": "USD",
    },
}
