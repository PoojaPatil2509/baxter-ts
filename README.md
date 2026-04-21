# baxter-ts

**AutoML time series library with BAX (Behavioural Analysis & eXplanation)**

[![PyPI version](https://img.shields.io/pypi/v/baxter-ts)](https://pypi.org/project/baxter-ts/)
[![Python](https://img.shields.io/pypi/pyversions/baxter-ts)](https://pypi.org/project/baxter-ts/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

One `fit()` call runs 10 preprocessing steps, trains 3 models in competition, picks the winner, explains *why* it made predictions in plain English, detects anomalies, and exports an interactive HTML report.

---

## Why baxter-ts?

| Pain today | What baxter-ts does |
|---|---|
| 60–70% of project time on preprocessing | 10 steps run automatically |
| Manually comparing model results | AutoML competition, winner auto-selected |
| "Why did the model predict this?" | BAX plain-English SHAP narrative |
| Anomalies found after damage is done | Every prediction gets an anomaly score |
| Rebuilding the same pipeline per dataset | Works on any time series domain or frequency |
| No audit trail | Full preprocessing + model log → HTML/PDF |

---

## Installation

```bash
pip install baxter-ts
```

---

## Quickstart (4 lines)

```python
from baxter_ts import BAXModel
import pandas as pd

df = pd.read_csv("your_data.csv")      # any time series CSV

model = BAXModel()
model.fit(df, target_col="sales", date_col="date")
model.predict(steps=30)                # 30-step future forecast
model.explain()                        # BAX narrative printed to stdout
model.anomalies()                      # anomaly DataFrame returned
model.visualize()                      # opens all Plotly charts
model.report("my_report")             # saves my_report.html
```

---

## Full API reference

### `BAXModel()`

```python
BAXModel(
    test_size       = 0.2,          # fraction held for test evaluation
    n_cv_splits     = 5,            # walk-forward CV folds
    outlier_treatment = "cap",      # "cap" (winsorise) or "flag" (boolean col)
    anomaly_method  = "ensemble",   # "ensemble" | "isolation_forest" | "zscore" | "iqr"
    contamination   = 0.05,         # expected anomaly fraction
    verbose         = True,         # print pipeline progress
)
```

### `.fit(df, target_col, date_col=None)`

Runs the full 10-step pipeline:

1. Datetime parsing and index validation
2. Frequency inference and gap detection
3. Missing value imputation (auto-selects: ffill / interpolation / seasonal mean / KNN)
4. Outlier detection and treatment (auto-selects: Z-score / IQR / Isolation Forest)
5. Stationarity testing (ADF + KPSS) and transformation (differencing, log, Box-Cox)
6. STL decomposition (trend + seasonal + residual components as features)
7. Scaling (auto-selects: MinMax / Standard / Robust)
8. Feature engineering (lags, rolling stats, EWM, calendar, Fourier terms, holidays)
9. Temporal train/test split (walk-forward, zero leakage)
10. AutoML competition: Random Forest vs XGBoost vs CatBoost → winner selected by MAE+RMSE+MAPE

### `.predict(steps=30) → pd.DataFrame`

Returns a DataFrame with `forecast` column indexed by future dates.

### `.explain() → str`

Prints and returns the BAX narrative: plain-English description of which features drove predictions and why, backed by SHAP values.

### `.anomalies() → pd.DataFrame`

Returns DataFrame with columns: `actual`, `predicted`, `residual`, `anomaly_flag`, `severity` (0/1/2), `severity_label`.

### `.scoreboard() → pd.DataFrame`

Returns the model competition table: MAE, RMSE, MAPE, R², composite score for each candidate.

### `.visualize(show=True) → dict`

Generates and optionally displays all interactive Plotly charts:
- Forecast with confidence bands
- Anomaly overlay
- SHAP feature importance (waterfall)
- Model competition scoreboard
- Residual analysis
- STL decomposition

### `.report(output_path) → str`

Saves a self-contained HTML report to `output_path.html`.
Includes all charts, metrics, BAX narrative, anomaly table, and preprocessing audit trail.

### `.summary() → dict`

Returns a flat dict of all key results for programmatic access.

---

## Supported data types

baxter-ts automatically adapts to any time series frequency:

| Frequency | Examples |
|---|---|
| Sub-minute | IoT sensor streams, tick data |
| Hourly | Energy consumption, web traffic |
| Daily | Sales, stock prices, weather |
| Weekly | Retail demand, marketing metrics |
| Monthly | Revenue, macroeconomic indicators |
| Quarterly | Financial reporting |
| Yearly | Annual statistics |

---

## Examples

### Stock price forecasting
```python
import yfinance as yf
from baxter_ts import BAXModel

df = yf.download("AAPL", start="2020-01-01", end="2024-01-01").reset_index()
df = df[["Date", "Close"]].rename(columns={"Date": "date", "Close": "price"})

model = BAXModel()
model.fit(df, target_col="price", date_col="date")
model.predict(steps=30)
model.report("aapl_forecast")
```

### IoT sensor anomaly detection
```python
import pandas as pd
from baxter_ts import BAXModel

df = pd.read_csv("sensor_data.csv")   # columns: timestamp, temperature

model = BAXModel(anomaly_method="ensemble", contamination=0.03)
model.fit(df, target_col="temperature", date_col="timestamp")
anom = model.anomalies()

print(anom[anom["severity_label"] == "anomaly"])
model.report("sensor_anomalies")
```

### Energy consumption
```python
from baxter_ts import BAXModel
import pandas as pd

df = pd.read_csv("energy.csv")  # hourly kWh readings

model = BAXModel(test_size=0.15, n_cv_splits=3)
model.fit(df, target_col="kwh", date_col="datetime")
forecast = model.predict(steps=168)  # one week ahead
model.explain()
model.visualize()
```

---

## Running the tests

```bash
# install dev dependencies
pip install baxter-ts[dev]

# run tests
pytest tests/ -v

# run with coverage
pytest tests/ -v --cov=baxter_ts --cov-report=term-missing
```

---

## Deploying your own build

```bash
# 1. Build the distribution
pip install build twine
python -m build

# 2. Upload to TestPyPI first
twine upload --repository testpypi dist/*

# 3. Test installation
pip install --index-url https://test.pypi.org/simple/ baxter-ts

# 4. Upload to real PyPI
twine upload dist/*
```

---

## Project structure

```
baxter-ts/
├── baxter_ts/
│   ├── core.py                  ← BAXModel (public API)
│   ├── preprocessing/
│   │   ├── validator.py         ← Steps 1+2
│   │   ├── imputer.py           ← Step 3
│   │   ├── outlier.py           ← Step 4
│   │   ├── transformer.py       ← Steps 5+6
│   │   ├── scaler.py            ← Step 7
│   │   ├── feature_eng.py       ← Step 8
│   │   └── splitter.py          ← Step 9
│   ├── models/
│   │   ├── base_model.py
│   │   ├── rf_model.py
│   │   ├── xgb_model.py
│   │   ├── catboost_model.py
│   │   └── selector.py          ← AutoML competition
│   ├── bax/
│   │   ├── explainer.py         ← SHAP
│   │   └── narrator.py          ← SHAP → plain English
│   ├── anomaly/
│   │   └── detector.py
│   ├── visualization/
│   │   └── plotter.py           ← Plotly charts
│   └── report/
│       └── generator.py         ← HTML report
├── tests/
│   └── test_baxter.py
├── pyproject.toml
└── README.md
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Value for engineers

> "baxter-ts turns a 2-week ML pipeline build into a 4-line script,
> while giving you more explainability and auditability than most
> hand-crafted solutions ever achieve."

- **Data engineers**: ingest raw CSV → production-ready forecast in minutes
- **ML engineers**: skip boilerplate preprocessing, focus on domain logic
- **Data scientists**: SHAP explanations built in, no extra code
- **DevOps / MLOps**: full audit trail exportable for compliance and review
- **Non-ML engineers**: human-readable BAX narrative means no ML knowledge needed to understand why the model predicted what it predicted
