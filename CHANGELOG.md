# Changelog

All notable changes to baxter-ts are documented here.

## [0.2.0] — 2026-07-06

The "trustworthy and deployable" release: forecasts you can read, intervals
you can act on, a baseline you must beat, and a model you can ship.

### Fixed
- **Forecasts are now in original units.** In v0.1.x, `predict()` returned
  values in the internal scaled/differenced/log space — a user forecasting
  prices near 150 got numbers like 0.02 whenever differencing was applied.
  The new `TargetInverter` walks the transform chain backwards
  (unscale → un-difference → expm1), anchored to the observed series.
- **Exact original-scale metrics.** MAE/RMSE/MAPE/R² are now computed on
  fully inverse-transformed predictions instead of the v0.1.3 std-ratio
  approximation (which is kept only as a fallback).
- **Frequency-alias bug:** the validator emitted pandas aliases `"min"`/`"h"`
  while the feature engineer expected `"T"`/`"H"`, so hourly and minutely
  data silently received *daily* lag defaults. All components now share
  `baxter_ts.utils.normalize_freq()`. Anchored aliases (`W-SUN`, `Q-DEC`)
  and multiples (`15min`) are handled too.
- Constant or near-constant targets no longer crash inside `adfuller`.
- Numeric columns can no longer be silently mis-detected as date columns.
- Removed the deprecated `infer_datetime_format` pandas argument.

### Added
- **Prediction intervals:** `predict()` now returns `forecast`, `lower`,
  `upper` columns. Split-conformal style: the half-width is the
  interval-level quantile of absolute test residuals, symmetric around the
  forecast, growing with `sqrt(horizon)` on differenced (integrated)
  series. Level is configurable via `BAXModel(interval=0.95)`. Charts and
  the HTML report draw the interval band.
- **SeasonalNaive baseline** joins the AutoML competition and appears on the
  scoreboard, so you can see whether ML actually beats "same value one
  season ago". The BAX narrative reports the winning margin — and explains
  the situation honestly if the baseline wins. Disable with
  `BAXModel(include_baseline=False)`.
- **Model persistence:** `model.save("model.joblib")` and
  `BAXModel.load(path)`. Everything needed for `predict()`, `explain()`,
  `anomalies()`, `visualize()` and `report()` survives the roundtrip.
- **Optional hyperparameter tuning:** `BAXModel(tune=True)` runs a small
  temporal random search per model (defaults always included, no new
  dependencies). Off by default so `fit()` stays fast.
- Anomaly tables now show actual/predicted/residual in original units.
- Numeric ID-like columns (strictly monotonic, all-unique integers — e.g.
  `batch_id`, auto-increment keys) are now auto-dropped like string IDs;
  they proxy the row index and invite overfitting.

### Changed / stability
- Lags and rolling windows are capped for short series instead of wiping
  out most rows with NaN.
- Walk-forward CV fold count adapts to the amount of training data.
- Clear `ValueError` (instead of an obscure crash) when fewer than 20
  usable rows are provided.
- Scoreboard metrics are displayed in original units when available.

## [0.1.3] — 2026-04-24
- Fixed MAPE/R² display: metrics were computed against differenced values;
  patched with a std-ratio conversion to original units.

## [0.1.2] — 2026-04-24
- Narrative polish: SHAP percentages rounded, preprocessing audit summary.

## [0.1.1] — 2026-04-24
- Initial public release: 10-step preprocessing, RF/XGBoost/CatBoost AutoML,
  SHAP-based BAX narrative, residual anomaly detection, offline HTML report.
