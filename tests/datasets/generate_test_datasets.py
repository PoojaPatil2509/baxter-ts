"""
Generate all test datasets for baxter-ts pre-deployment validation.
Run once to create CSV files, then run test_all_datasets.py to validate.

Usage:
    python tests/datasets/generate_test_datasets.py
"""

import numpy as np
import pandas as pd
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__))
np.random.seed(42)


def save(df: pd.DataFrame, name: str):
    path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {name}.csv  ({df.shape[0]} rows x {df.shape[1]} cols) — dtypes: {dict(df.dtypes)}")
    return path


print("Generating test datasets...\n")

# ── 1. Retail sales — daily, single target, clean ─────────────────────
n = 730
dates = pd.date_range("2020-01-01", periods=n, freq="D")
sales = (
    500
    + np.linspace(0, 100, n)
    + 80 * np.sin(2 * np.pi * np.arange(n) / 7)
    + 50 * np.sin(2 * np.pi * np.arange(n) / 365)
    + np.random.randn(n) * 20
)
save(pd.DataFrame({"date": dates, "sales": sales}), "01_retail_sales_daily")

# ── 2. Retail sales + exogenous — categorical + numeric extra cols ────
df2 = pd.DataFrame({
    "date":        dates,
    "sales":       sales,
    "temperature": 15 + 12 * np.sin(2 * np.pi * np.arange(n) / 365) + np.random.randn(n) * 2,
    "promo":       np.random.choice([0, 1], n, p=[0.75, 0.25]),
    "day_type":    np.where(pd.DatetimeIndex(dates).dayofweek < 5, "weekday", "weekend"),
    "store_region":np.random.choice(["North", "South", "East", "West"], n),
    "price":       9.99 + np.random.randn(n) * 0.5,
    "competitor_price": 10.50 + np.random.randn(n) * 0.8,
})
# inject some missing in extra cols (not target)
df2.loc[np.random.choice(n, 30, replace=False), "temperature"] = np.nan
df2.loc[np.random.choice(n, 15, replace=False), "price"] = np.nan
save(df2, "02_retail_multivariate_with_categoricals")

# ── 3. Energy (IoT) — hourly, high frequency ──────────────────────────
n3 = 8760  # 1 year hourly
dates3 = pd.date_range("2022-01-01", periods=n3, freq="h")
kwh = (
    50
    + 20 * np.sin(2 * np.pi * np.arange(n3) / 24)
    + 10 * np.sin(2 * np.pi * np.arange(n3) / (24 * 7))
    + np.random.randn(n3) * 3
)
kwh[kwh < 0] = 0
# Inject anomaly spikes
kwh[1200:1205] *= 8
df3 = pd.DataFrame({
    "timestamp": dates3,
    "kwh":       kwh,
    "outdoor_temp": 10 + 15 * np.sin(2 * np.pi * np.arange(n3) / (24 * 365)) + np.random.randn(n3),
    "is_holiday": np.random.choice([0, 1], n3, p=[0.97, 0.03]),
})
save(df3, "03_energy_hourly_iot")

# ── 4. Stock price — daily, high volatility, skewed ───────────────────
n4 = 1000
dates4 = pd.date_range("2019-01-01", periods=n4, freq="B")  # business days
log_returns = np.random.randn(n4) * 0.015
prices = 100 * np.exp(np.cumsum(log_returns))
df4 = pd.DataFrame({
    "date":   dates4,
    "close":  prices,
    "volume": np.abs(np.random.randn(n4) * 1e6 + 5e6),
    "high":   prices * (1 + np.abs(np.random.randn(n4) * 0.01)),
    "low":    prices * (1 - np.abs(np.random.randn(n4) * 0.01)),
})
save(df4, "04_stock_price_daily")

# ── 5. Website traffic — monthly, short series ───────────────────────
n5 = 48  # 4 years monthly
dates5 = pd.date_range("2020-01-01", periods=n5, freq="MS")
visits = (
    10000
    + np.linspace(0, 5000, n5)
    + 2000 * np.sin(2 * np.pi * np.arange(n5) / 12)
    + np.random.randn(n5) * 500
)
df5 = pd.DataFrame({
    "month":        dates5,
    "page_views":   np.abs(visits).astype(int),
    "sessions":     np.abs(visits * 0.7 + np.random.randn(n5) * 200).astype(int),
    "bounce_rate":  np.clip(0.4 + np.random.randn(n5) * 0.05, 0.2, 0.8),
    "channel":      np.random.choice(["organic", "paid", "social", "direct"], n5),
})
save(df5, "05_web_traffic_monthly")

# ── 6. Manufacturing — with missing values in target ──────────────────
n6 = 500
dates6 = pd.date_range("2021-06-01", periods=n6, freq="D")
defect_rate = np.abs(
    0.05
    + 0.02 * np.sin(2 * np.pi * np.arange(n6) / 30)
    + np.random.randn(n6) * 0.005
)
# Heavy missing: 15% missing in target
miss_idx = np.random.choice(n6, int(n6 * 0.15), replace=False)
defect_rate_with_nan = defect_rate.copy().astype(float)
defect_rate_with_nan[miss_idx] = np.nan
df6 = pd.DataFrame({
    "date":          dates6,
    "defect_rate":   defect_rate_with_nan,
    "machine_speed": 1000 + np.random.randn(n6) * 50,
    "temperature":   75 + np.random.randn(n6) * 5,
    "shift":         np.tile(["morning", "afternoon", "night"], n6 // 3 + 1)[:n6],
    "batch_id":      np.arange(n6),   # ID-like column — should be auto-dropped
})
save(df6, "06_manufacturing_heavy_missing")

# ── 7. Healthcare — weekly, long gap periods ──────────────────────────
n7 = 260  # 5 years weekly
dates7 = pd.date_range("2018-01-01", periods=n7, freq="W")
admissions = np.abs(
    200
    + 30 * np.sin(2 * np.pi * np.arange(n7) / 52)
    + np.random.randn(n7) * 15
).astype(int)
# Introduce large gaps (simulate hospital closure weeks)
admissions = admissions.astype(float)
admissions[50:55] = np.nan  # closure
admissions[130:133] = np.nan
df7 = pd.DataFrame({
    "week_start":    dates7,
    "admissions":    admissions,
    "flu_index":     np.abs(np.random.randn(n7) * 10 + 50),
    "beds_available":np.random.randint(80, 120, n7),
    "season":        pd.cut(
        pd.DatetimeIndex(dates7).month,
        bins=[0, 3, 6, 9, 12],
        labels=["winter", "spring", "summer", "autumn"]
    ).astype(str),
})
save(df7, "07_healthcare_weekly_gaps")

# ── 8. Finance — quarterly, very short ───────────────────────────────
n8 = 32  # 8 years quarterly
dates8 = pd.date_range("2016-01-01", periods=n8, freq="QS")
revenue = 1e6 * (
    1
    + np.linspace(0, 0.5, n8)
    + 0.1 * np.sin(2 * np.pi * np.arange(n8) / 4)
    + np.random.randn(n8) * 0.05
)
df8 = pd.DataFrame({
    "quarter":    dates8,
    "revenue":    revenue,
    "expenses":   revenue * (0.7 + np.random.randn(n8) * 0.05),
    "headcount":  np.random.randint(100, 300, n8),
    "region":     np.random.choice(["APAC", "EMEA", "AMER"], n8),
})
save(df8, "08_finance_quarterly_short")

# ── 9. Date already set as index ─────────────────────────────────────
n9 = 400
idx9 = pd.date_range("2020-01-01", periods=n9, freq="D")
df9 = pd.DataFrame(
    {
        "demand":  500 + np.linspace(0, 100, n9) + np.random.randn(n9) * 20,
        "weather": np.random.choice(["sunny", "rainy", "cloudy"], n9),
    },
    index=idx9,
)
df9.index.name = "date"
save(df9.reset_index(), "09_demand_date_as_index")

# ── 10. Extreme edge cases — constant cols, all-NaN col, bool col ─────
n10 = 300
dates10 = pd.date_range("2023-01-01", periods=n10, freq="D")
target10 = 100 + np.linspace(0, 50, n10) + np.random.randn(n10) * 5
df10 = pd.DataFrame({
    "date":           dates10,
    "value":          target10,
    "constant_col":   np.ones(n10),               # should be dropped
    "all_nan_col":    np.full(n10, np.nan),         # should be dropped
    "bool_feature":   np.random.choice([True, False], n10),  # should → int
    "negative_vals":  np.random.randn(n10) * 10,   # negative values ok
    "big_outliers":   np.where(
        np.random.rand(n10) < 0.02,
        target10 * 100,  # extreme outliers
        target10
    ),
})
save(df10, "10_edge_cases_constant_allnan_bool")

print(f"\nAll 10 datasets saved to: {OUTPUT_DIR}")
