Open `D:\baxter-ts\README.md`, press `Ctrl+A`, delete, paste this entire content:

---

```markdown
# baxter-ts

> AutoML time series library with BAX (Behavioural Analysis & eXplanation)

[![PyPI version](https://img.shields.io/pypi/v/baxter-ts)](https://pypi.org/project/baxter-ts/)
[![Python](https://img.shields.io/pypi/pyversions/baxter-ts)](https://pypi.org/project/baxter-ts/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![TestPyPI](https://img.shields.io/badge/TestPyPI-passing-brightgreen)](https://test.pypi.org/project/baxter-ts/)

**baxter-ts** is a one-call AutoML pipeline for any time series data. It automatically preprocesses your data, trains and compares three models, selects the best one, explains every prediction in plain English using SHAP (the BAX layer), detects anomalies, and produces a fully interactive offline HTML report — all from a single `fit()` call.

No manual preprocessing. No model tuning. No explainability code. Just results.

---

## Contents

- [Why baxter-ts](#why-baxter-ts)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [How it works](#how-it-works)
- [API reference](#api-reference)
- [Supported data types](#supported-data-types)
- [Examples](#examples)
- [Report output](#report-output)
- [Project structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Why baxter-ts

| Problem today | What baxter-ts solves |
|---|---|
| 60–70% of project time spent on preprocessing | 10 preprocessing steps run automatically |
| Manually training and comparing multiple models | AutoML competition — winner selected automatically |
| Client asks "why did the model predict this?" | BAX generates a plain-English SHAP narrative |
| Anomalies discovered only after damage is done | Every prediction carries an anomaly score |
| Rebuilding the same pipeline for every new dataset | One API works on stock prices, IoT, sales, energy — any time series |
| No audit trail for model decisions | Full preprocessing and model log exported to HTML |

---

## Installation

```bash
pip install baxter-ts
```

Requires Python 3.9 or higher.

---

## Quickstart

```python
from baxter_ts import BAXModel
import pandas as pd

df = pd.read_csv("your_data.csv")

model = BAXModel()
model.fit(df, target_col="sales", date_col="date")

model.predict(steps=30)     # 30-step future forecast
model.explain()             # BAX plain-English narrative
model.anomalies()           # anomaly DataFrame
model.visualize()           # 7 interactive Plotly charts
model.report("my_report")   # saves my_report.html — open in any browser
```

---

## How it works

Every `fit()` call runs a 10-step pipeline automatically:

```
Raw data
    │
    ├── Step 1    Datetime parsing       auto-detects date column, parses any format
    ├── Step 2    Frequency inference    detects minutely / hourly / daily / weekly / monthly
    ├── Step 3    Missing value fill     auto selects ffill / interpolation / seasonal mean / KNN
    ├── Step 4    Outlier handling       auto selects Z-score / IQR / Isolation Forest → cap or flag
    ├── Step 5    Stationarity          ADF + KPSS test, applies differencing or log transform
    ├── Step 6    STL decomposition     extracts trend, seasonal, residual as model features
    ├── Step 7    Scaling               auto selects MinMax / Standard / Robust
    ├── Step 8    Feature engineering   lags, rolling stats, EWM, Fourier terms, calendar, holidays
    ├── Step 9    Temporal split        walk-forward CV, zero data leakage
    │
    ├── AutoML competition
    │       ├── Random Forest
    │       ├── XGBoost
    │       └── CatBoost
    │             → winner selected by composite MAE + RMSE + MAPE score
    │
    ├── BAX explanation      SHAP values translated to plain English narrative
    ├── Anomaly detection    ensemble of Isolation Forest + Z-score + IQR on residuals
    └── HTML report          7 interactive Plotly charts, metrics, audit trail — fully offline
```

---

## API reference

### `BAXModel`

```python
BAXModel(
    test_size         = 0.2,        # fraction of data held out for evaluation
    n_cv_splits       = 5,          # number of walk-forward cross-validation folds
    outlier_treatment = "cap",      # "cap" (winsorise) or "flag" (add boolean column)
    anomaly_method    = "ensemble", # "ensemble" | "isolation_forest" | "zscore" | "iqr"
    contamination     = 0.05,       # expected anomaly fraction for Isolation Forest
    verbose           = True,       # print pipeline progress to console
)
```

---

### `.fit(df, target_col, date_col=None)`

Runs the full preprocessing and AutoML pipeline.

| Parameter | Type | Description |
|---|---|---|
| `df` | `pd.DataFrame` | Raw time series data |
| `target_col` | `str` | Name of the column to forecast |
| `date_col` | `str` or `None` | Datetime column name. Auto-detected if not provided |

Returns `self` — supports method chaining.

---

### `.predict(steps=30) → pd.DataFrame`

Generates a future forecast for `steps` time periods ahead.

Returns a DataFrame indexed by future dates with a single `forecast` column.

---

### `.explain() → str`

Prints and returns the BAX behavioural narrative.

Plain-English description of which features drove the model's predictions and by how much, backed by SHAP (SHapley Additive eXplanations) values. Non-technical stakeholders can read this directly without any ML background.

---

### `.anomalies() → pd.DataFrame`

Runs anomaly detection on model residuals (actual minus predicted). Detecting anomalies on residuals — not raw values — catches unexpected deviations the model itself did not predict, which is far more useful than flagging obvious spikes.

Returns a DataFrame with columns:

| Column | Description |
|---|---|
| `actual` | True observed value |
| `predicted` | Model prediction |
| `residual` | Difference between actual and predicted |
| `anomaly_flag` | 1 = anomaly, 0 = normal |
| `severity` | 0 = normal, 1 = suspicious, 2 = anomaly |
| `severity_label` | `"normal"`, `"suspicious"`, or `"anomaly"` |

---

### `.scoreboard() → pd.DataFrame`

Returns the full AutoML competition results with MAE, RMSE, MAPE, R², CV scores, and composite score for all three candidate models.

---

### `.visualize(show=True) → dict`

Generates and optionally displays all 7 interactive Plotly charts. Returns a dictionary of figure objects for further customisation.

Charts included:

| Chart | What it shows |
|---|---|
| Forecast | Actual vs predicted on test set + future forecast line |
| Anomaly overlay | Flagged anomaly and suspicious points on the series |
| SHAP importance | Top features ranked by mean absolute SHAP value |
| Model scoreboard | MAE, RMSE, MAPE bars for all 3 models side by side |
| Residuals over time | Scatter plot coloured by anomaly severity |
| Residual distribution | Histogram showing error distribution shape |
| STL decomposition | Trend, seasonal, and residual component panels |

---

### `.report(output_path) → str`

Saves a fully self-contained HTML report to `output_path.html`.

The Plotly JS bundle (4.7 MB) is embedded inline inside the file — no internet connection is needed to open it. Works in Chrome, Firefox, Edge, and Safari. Includes all 7 charts, model metrics, BAX narrative, top anomaly timestamps, and a full preprocessing audit trail showing every transformation applied with its parameters.

---

### `.summary() → dict`

Returns a flat dictionary of all key results for programmatic access or logging.

```python
model.summary()
# {
#     "target_col":        "sales",
#     "frequency":         "D",
#     "best_model":        "CatBoost",
#     "test_mae":          0.1367,
#     "test_rmse":         0.203,
#     "test_mape":         70.9,
#     "test_r2":           0.9523,
#     "anomalies_found":   2,
#     "shap_top_features": ["seasonal_component", "residual_component", "dayofweek"],
#     "train_rows":        400,
#     "test_rows":         100,
# }
```

---

## Supported data types

baxter-ts automatically detects the frequency of your data and adapts all preprocessing, lag generation, rolling windows, Fourier terms, and calendar features accordingly.

| Frequency | Typical use cases |
|---|---|
| Sub-minute (1 min) | IoT sensor streams, industrial equipment monitoring |
| Hourly | Energy consumption, weather stations, web traffic |
| Daily | Retail sales, stock prices, hospital admissions |
| Weekly | Demand forecasting, marketing performance |
| Monthly | Revenue, macroeconomic indicators, subscriptions |
| Quarterly | Financial reporting, EPS, budget cycles |
| Yearly | Annual statistics, long-range planning |

Data quality issues handled automatically:

- Missing values up to 30% of the series
- Extreme outlier spikes
- Unsorted or duplicate timestamps
- Sudden level shifts
- Non-stationary series (random walks, strong trends)
- Negative values and integer-only count data
- Mixed-frequency gaps (e.g. trading data with weekend gaps)

---

## Examples

### Daily retail sales

```python
from baxter_ts import BAXModel
import pandas as pd

df = pd.read_csv("sales.csv")

model = BAXModel()
model.fit(df, target_col="sales", date_col="date")

forecast = model.predict(steps=30)
print(forecast)

model.explain()
model.report("sales_report")
```

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

df = pd.read_csv("sensor_data.csv")

model = BAXModel(anomaly_method="ensemble", contamination=0.03)
model.fit(df, target_col="temperature", date_col="timestamp")

anom = model.anomalies()
print(anom[anom["severity_label"] == "anomaly"])
model.report("sensor_report")
```

### Hourly energy consumption

```python
from baxter_ts import BAXModel
import pandas as pd

df = pd.read_csv("energy.csv")

model = BAXModel(test_size=0.15, n_cv_splits=3)
model.fit(df, target_col="kwh", date_col="datetime")

forecast = model.predict(steps=168)   # 7 days ahead
model.explain()
model.visualize()
```

### Monthly revenue with full output

```python
from baxter_ts import BAXModel
import pandas as pd

df = pd.read_csv("revenue.csv")

model = BAXModel(n_cv_splits=3, anomaly_method="zscore")
model.fit(df, target_col="revenue", date_col="month")

forecast  = model.predict(steps=12)
anomalies = model.anomalies()
narrative = model.explain()
scoreboard = model.scoreboard()
summary   = model.summary()

model.report("revenue_report")

print(f"Winner : {summary['best_model']}")
print(f"R²     : {summary['test_r2']}")
print(f"Anomalies found: {summary['anomalies_found']}")
```

---

## Report output

Every `.report()` call produces a single self-contained HTML file.

| Section | Contents |
|---|---|
| Model performance | MAE, RMSE, MAPE, R² metric cards |
| AutoML scoreboard | All 3 models with CV scores and composite rank |
| BAX explanation | Plain-English SHAP narrative with preprocessing summary |
| Anomaly summary | Total points, anomaly count, suspicious count, top timestamps |
| Preprocessing audit | Every step applied with method, parameters, and row counts |
| Forecast chart | Actual, predicted, and future forecast on one interactive plot |
| Anomaly overlay | Series with anomaly and suspicious points highlighted |
| SHAP importance | Horizontal bar chart of top features by influence percentage |
| Model scoreboard | Side-by-side MAE, RMSE, MAPE bars for all models |
| Residual analysis | Scatter over time + histogram — coloured by severity |
| STL decomposition | Trend, seasonal, and residual panels with hover tooltips |

The report has no external dependencies and works fully offline.

---

## Project structure

```
baxter-ts/
├── baxter_ts/
│   ├── core.py                      ← BAXModel — public API entry point
│   ├── preprocessing/
│   │   ├── validator.py             ← datetime parsing and frequency inference
│   │   ├── imputer.py               ← missing value imputation
│   │   ├── outlier.py               ← outlier detection and treatment
│   │   ├── transformer.py           ← stationarity testing and STL decomposition
│   │   ├── scaler.py                ← feature scaling
│   │   ├── feature_eng.py           ← lag, rolling, Fourier, and calendar features
│   │   ├── splitter.py              ← temporal train/test split
│   │   └── column_handler.py        ← categorical encoding and column cleanup
│   ├── models/
│   │   ├── base_model.py            ← shared fit, score, and CV logic
│   │   ├── rf_model.py              ← Random Forest
│   │   ├── xgb_model.py             ← XGBoost
│   │   ├── catboost_model.py        ← CatBoost
│   │   └── selector.py              ← AutoML competition and winner selection
│   ├── bax/
│   │   ├── explainer.py             ← SHAP TreeExplainer wrapper
│   │   └── narrator.py              ← SHAP values to plain-English narrative
│   ├── anomaly/
│   │   └── detector.py              ← ensemble anomaly detection on residuals
│   ├── visualization/
│   │   └── plotter.py               ← 7 interactive Plotly charts
│   └── report/
│       └── generator.py             ← self-contained HTML report generator
├── tests/
│   ├── test_baxter.py               ← 37 unit tests
│   ├── test_comprehensive.py        ← 30 scenario tests across domains and frequencies
│   ├── test_all_datasets.py         ← 10 real-world CSV dataset tests
│   └── datasets/                    ← 10 test CSV files
├── examples/
│   ├── quickstart.py                ← runnable end-to-end example
│   └── demo_report.html             ← sample output report
├── pre_launch_check.py              ← pre-publish verification script
├── CONTRIBUTING.md                  ← development setup and release process
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## Contributing

Contributions, bug reports, and feature requests are welcome.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, running the test suite, and the release process.

To report a bug or request a feature, open an issue at:
https://github.com/PoojaPatil2509/baxter-ts/issues

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
