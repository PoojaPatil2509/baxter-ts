"""
baxter-ts quickstart example.
Run this file directly: python examples/quickstart.py
"""

import numpy as np
import pandas as pd
from baxter_ts import BAXModel

# ── Generate synthetic sales data ────────────────────────────────────
np.random.seed(42)
n = 500
dates  = pd.date_range("2021-01-01", periods=n, freq="D")
trend  = np.linspace(200, 400, n)
weekly = 30 * np.sin(2 * np.pi * np.arange(n) / 7)
yearly = 50 * np.sin(2 * np.pi * np.arange(n) / 365)
noise  = np.random.randn(n) * 15

values = trend + weekly + yearly + noise

# Inject missing values
miss_idx = np.random.choice(n, size=15, replace=False)
values[miss_idx] = np.nan

df = pd.DataFrame({"date": dates, "sales": values})

print("Sample data:")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"Missing values: {df['sales'].isna().sum()}\n")

# ── Fit ───────────────────────────────────────────────────────────────
model = BAXModel(
    test_size=0.2,
    n_cv_splits=3,
    outlier_treatment="cap",
    anomaly_method="ensemble",
    verbose=True,
)
model.fit(df, target_col="sales", date_col="date")

# ── Forecast ──────────────────────────────────────────────────────────
forecast = model.predict(steps=30)
print("\n30-day forecast:")
print(forecast.head(10))

# ── BAX Explanation ───────────────────────────────────────────────────
model.explain()

# ── Anomalies ─────────────────────────────────────────────────────────
anom_df = model.anomalies()
print("\nAnomaly summary:")
print(anom_df["severity_label"].value_counts())

# ── Scoreboard ────────────────────────────────────────────────────────
print("\nModel scoreboard:")
print(model.scoreboard())

# ── Summary dict ──────────────────────────────────────────────────────
print("\nSummary:")
for k, v in model.summary().items():
    print(f"  {k}: {v}")

# ── Report ────────────────────────────────────────────────────────────
report_path = model.report("quickstart_report")
print(f"\nReport saved to: {report_path}")
print("Open it in your browser to see all interactive charts.")
