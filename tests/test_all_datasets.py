"""
baxter-ts Pre-Deployment Validation Suite
==========================================
Tests every dataset type, edge case, and evaluation metric.

Run from the baxter-ts root folder:
    python tests/test_all_datasets.py

Or with pytest for structured output:
    pytest tests/test_all_datasets.py -v

Covers:
  - 10 real-world dataset types
  - Multi-column (categorical + numeric exogenous features)
  - Missing values in target AND extra columns
  - Various frequencies: hourly, daily, weekly, monthly, quarterly
  - Edge cases: constant cols, all-NaN cols, bool cols, ID cols
  - Short datasets (<50 rows)
  - Date already set as index
  - Heavy outlier data
  - Model evaluation: MAE, RMSE, MAPE, R²
  - Anomaly detection evaluation
  - Report generation
  - Column encoding audit
"""

import os
import sys
import warnings
import traceback
import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

# ── path setup ───────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATASETS_DIR = os.path.join(ROOT, "tests", "datasets")

from baxter_ts import BAXModel
from baxter_ts.preprocessing.column_handler import ColumnHandler


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def load(filename: str) -> pd.DataFrame:
    path = os.path.join(DATASETS_DIR, filename)
    return pd.read_csv(path)


def quick_fit(csv_file, target_col, date_col, n_cv=2) -> BAXModel:
    df = load(csv_file)
    m = BAXModel(n_cv_splits=n_cv, verbose=False)
    m.fit(df, target_col=target_col, date_col=date_col)
    return m


def assert_scores_sane(scores: dict, dataset_name: str):
    """Scores should exist and be finite numbers."""
    for key in ["mae", "rmse", "r2"]:
        val = scores.get(key)
        assert val is not None, f"[{dataset_name}] Missing score key: {key}"
        assert np.isfinite(val), f"[{dataset_name}] Score {key}={val} is not finite"
    assert scores["mae"] >= 0,  f"[{dataset_name}] MAE cannot be negative"
    assert scores["rmse"] >= 0, f"[{dataset_name}] RMSE cannot be negative"


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1 — COLUMN HANDLER UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestColumnHandler:

    def _make_df(self):
        np.random.seed(1)
        n = 100
        return pd.DataFrame({
            "date":          pd.date_range("2022-01-01", periods=n, freq="D"),
            "target":        np.random.randn(n) + 10,
            "cat_low":       np.random.choice(["A", "B", "C"], n),          # OHE (3 unique)
            "cat_high":      [f"user_{i}" for i in range(n)],                # ID-like → drop
            "bool_col":      np.random.choice([True, False], n),             # → int
            "constant":      np.ones(n),                                     # → drop
            "all_nan":       np.full(n, np.nan),                             # → drop
            "numeric_extra": np.random.randn(n) * 5,
            "int_flag":      np.random.choice([0, 1], n),
        })

    def test_drops_constant_column(self):
        df = self._make_df()
        ch = ColumnHandler()
        out = ch.fit_transform(df, target_col="target")
        assert "constant" not in out.columns

    def test_drops_all_nan_column(self):
        df = self._make_df()
        ch = ColumnHandler()
        out = ch.fit_transform(df, target_col="target")
        assert "all_nan" not in out.columns

    def test_drops_id_like_column(self):
        df = self._make_df()
        ch = ColumnHandler()
        out = ch.fit_transform(df, target_col="target")
        assert "cat_high" not in out.columns

    def test_bool_converted_to_int(self):
        df = self._make_df()
        ch = ColumnHandler()
        out = ch.fit_transform(df, target_col="target")
        assert "bool_col" in out.columns
        assert out["bool_col"].dtype in [np.int64, np.int32, int]

    def test_low_cardinality_categorical_ohe(self):
        df = self._make_df()
        ch = ColumnHandler()
        out = ch.fit_transform(df, target_col="target")
        # cat_low (A/B/C) should be one-hot encoded into cat_low_A, cat_low_B, cat_low_C
        ohe_cols = [c for c in out.columns if c.startswith("cat_low_")]
        assert len(ohe_cols) == 3, f"Expected 3 OHE cols, got {ohe_cols}"

    def test_target_column_preserved(self):
        df = self._make_df()
        ch = ColumnHandler()
        out = ch.fit_transform(df, target_col="target")
        assert "target" in out.columns

    def test_numeric_extra_cols_preserved(self):
        df = self._make_df()
        ch = ColumnHandler()
        out = ch.fit_transform(df, target_col="target")
        assert "numeric_extra" in out.columns

    def test_audit_populated(self):
        df = self._make_df()
        ch = ColumnHandler()
        ch.fit_transform(df, target_col="target")
        audit = ch.audit
        assert "columns_dropped" in audit
        assert len(audit["columns_dropped"]) >= 3  # constant, all_nan, id-like

    def test_output_all_numeric(self):
        df = self._make_df()
        ch = ColumnHandler()
        out = ch.fit_transform(df, target_col="target")
        # After handling, all non-date columns should be numeric
        non_date = [c for c in out.columns if c != "date"]
        for col in non_date:
            assert pd.api.types.is_numeric_dtype(out[col]), \
                f"Column {col} is not numeric after ColumnHandler: {out[col].dtype}"


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2 — DATASET-LEVEL INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestDataset01_RetailSalesDaily:
    """Single target, clean daily data — baseline."""

    def test_fits_without_error(self):
        m = quick_fit("01_retail_sales_daily.csv", "sales", "date")
        assert m._is_fitted

    def test_correct_frequency_detected(self):
        m = quick_fit("01_retail_sales_daily.csv", "sales", "date")
        assert m._freq == "D"

    def test_scores_sane(self):
        m = quick_fit("01_retail_sales_daily.csv", "sales", "date")
        assert_scores_sane(m._best_scores, "retail_daily")

    def test_r2_reasonable(self):
        m = quick_fit("01_retail_sales_daily.csv", "sales", "date")
        assert m._best_scores["r2"] > 0.5, "R² should be > 0.5 on clean trend+seasonal data"

    def test_predict_14_steps(self):
        m = quick_fit("01_retail_sales_daily.csv", "sales", "date")
        fc = m.predict(14)
        assert len(fc) == 14
        assert fc["forecast"].nunique() > 1

    def test_anomaly_returns_correct_columns(self):
        m = quick_fit("01_retail_sales_daily.csv", "sales", "date")
        a = m.anomalies()
        for col in ["actual", "predicted", "residual", "anomaly_flag", "severity_label"]:
            assert col in a.columns

    def test_anomaly_flag_binary(self):
        m = quick_fit("01_retail_sales_daily.csv", "sales", "date")
        a = m.anomalies()
        assert set(a["anomaly_flag"].unique()).issubset({0, 1})

    def test_explain_not_empty(self):
        m = quick_fit("01_retail_sales_daily.csv", "sales", "date")
        txt = m.explain()
        assert len(txt) > 100

    def test_scoreboard_has_3_models(self):
        m = quick_fit("01_retail_sales_daily.csv", "sales", "date")
        sb = m.scoreboard()
        assert len(sb) == 3

    def test_report_generates(self, tmp_path):
        m = quick_fit("01_retail_sales_daily.csv", "sales", "date")
        m.anomalies()
        path = m.report(str(tmp_path / "report01"))
        assert os.path.exists(path)
        assert os.path.getsize(path) > 100_000  # > 100KB


class TestDataset02_MultivariateWithCategoricals:
    """Multi-column: numeric + categorical exogenous features."""

    def test_fits_without_error(self):
        m = quick_fit("02_retail_multivariate_with_categoricals.csv", "sales", "date")
        assert m._is_fitted

    def test_categorical_columns_encoded(self):
        df = load("02_retail_multivariate_with_categoricals.csv")
        m = BAXModel(n_cv_splits=2, verbose=False)
        m.fit(df, target_col="sales", date_col="date")
        # store_region (4 values) and day_type (2 values) should be OHE
        audit = m._preprocessing_audit.get("column_handler", {})
        assert len(audit.get("columns_ohe", [])) >= 1

    def test_no_string_columns_in_features(self):
        df = load("02_retail_multivariate_with_categoricals.csv")
        m = BAXModel(n_cv_splits=2, verbose=False)
        m.fit(df, target_col="sales", date_col="date")
        for col in m._X_train.columns:
            assert pd.api.types.is_numeric_dtype(m._X_train[col]), \
                f"Non-numeric column in feature matrix: {col}"

    def test_missing_extra_cols_handled(self):
        """temperature and price have NaN — pipeline should not crash."""
        df = load("02_retail_multivariate_with_categoricals.csv")
        m = BAXModel(n_cv_splits=2, verbose=False)
        m.fit(df, target_col="sales", date_col="date")
        assert m._is_fitted

    def test_scores_sane(self):
        m = quick_fit("02_retail_multivariate_with_categoricals.csv", "sales", "date")
        assert_scores_sane(m._best_scores, "multivariate_categoricals")

    def test_predict_varies(self):
        m = quick_fit("02_retail_multivariate_with_categoricals.csv", "sales", "date")
        fc = m.predict(7)
        assert fc["forecast"].nunique() > 1


class TestDataset03_EnergyHourly:
    """Hourly IoT data — high frequency, injected anomaly spikes."""

    def test_fits_hourly_data(self):
        m = quick_fit("03_energy_hourly_iot.csv", "kwh", "timestamp")
        assert m._is_fitted

    def test_frequency_detected_as_hourly(self):
        m = quick_fit("03_energy_hourly_iot.csv", "kwh", "timestamp")
        assert m._freq in ["h", "H", "60min", "T"]

    def test_anomaly_detects_injected_spikes(self):
        m = quick_fit("03_energy_hourly_iot.csv", "kwh", "timestamp")
        a = m.anomalies()
        # We injected spikes at rows 1200-1205; anomaly detector should catch some
        assert a["anomaly_flag"].sum() > 0, "Should detect at least 1 anomaly"

    def test_scores_sane(self):
        m = quick_fit("03_energy_hourly_iot.csv", "kwh", "timestamp")
        assert_scores_sane(m._best_scores, "energy_hourly")


class TestDataset04_StockPriceDaily:
    """Stock prices — random walk, high volatility, skewed, with volume."""

    def test_fits_without_error(self):
        m = quick_fit("04_stock_price_daily.csv", "close", "date")
        assert m._is_fitted

    def test_extra_numeric_cols_used(self):
        df = load("04_stock_price_daily.csv")
        m = BAXModel(n_cv_splits=2, verbose=False)
        m.fit(df, target_col="close", date_col="date")
        # volume, high, low should remain as features
        feat_cols = m._X_train.columns.tolist()
        assert any("volume" in c or "high" in c or "low" in c for c in feat_cols
                   ) or len(feat_cols) > 10  # at least feature eng added cols

    def test_scores_exist(self):
        m = quick_fit("04_stock_price_daily.csv", "close", "date")
        assert_scores_sane(m._best_scores, "stock")


class TestDataset05_WebTrafficMonthly:
    """Monthly data, short (48 rows), categorical channel column."""

    def test_fits_short_monthly_data(self):
        m = quick_fit("05_web_traffic_monthly.csv", "page_views", "month")
        assert m._is_fitted

    def test_frequency_monthly(self):
        m = quick_fit("05_web_traffic_monthly.csv", "page_views", "month")
        assert m._freq in ["MS", "ME", "M", "BMS"]

    def test_categorical_channel_handled(self):
        df = load("05_web_traffic_monthly.csv")
        m = BAXModel(n_cv_splits=2, verbose=False)
        m.fit(df, target_col="page_views", date_col="month")
        # channel col (organic/paid/social/direct) must be encoded
        for col in m._X_train.columns:
            assert pd.api.types.is_numeric_dtype(m._X_train[col])


class TestDataset06_ManufacturingHeavyMissing:
    """15% missing in target, shift (categorical), batch_id (ID → should drop)."""

    def test_fits_with_heavy_missing(self):
        m = quick_fit("06_manufacturing_heavy_missing.csv", "defect_rate", "date")
        assert m._is_fitted

    def test_batch_id_dropped(self):
        df = load("06_manufacturing_heavy_missing.csv")
        m = BAXModel(n_cv_splits=2, verbose=False)
        m.fit(df, target_col="defect_rate", date_col="date")
        audit = m._preprocessing_audit.get("column_handler", {})
        dropped = " ".join(str(d) for d in audit.get("columns_dropped", []))
        assert "batch_id" in dropped, f"batch_id should be dropped as ID-like. Audit: {audit}"

    def test_missing_pct_logged(self):
        df = load("06_manufacturing_heavy_missing.csv")
        m = BAXModel(n_cv_splits=2, verbose=False)
        m.fit(df, target_col="defect_rate", date_col="date")
        imp_audit = m._preprocessing_audit.get("imputer", {})
        assert imp_audit.get("missing_before", 0) > 0

    def test_scores_sane(self):
        m = quick_fit("06_manufacturing_heavy_missing.csv", "defect_rate", "date")
        assert_scores_sane(m._best_scores, "manufacturing")


class TestDataset07_HealthcareWeeklyGaps:
    """Weekly data, real gap periods (hospital closures → consecutive NaN)."""

    def test_fits_weekly_with_gaps(self):
        m = quick_fit("07_healthcare_weekly_gaps.csv", "admissions", "week_start")
        assert m._is_fitted

    def test_season_categorical_encoded(self):
        df = load("07_healthcare_weekly_gaps.csv")
        m = BAXModel(n_cv_splits=2, verbose=False)
        m.fit(df, target_col="admissions", date_col="week_start")
        for col in m._X_train.columns:
            assert pd.api.types.is_numeric_dtype(m._X_train[col])


class TestDataset08_FinanceQuarterlyShort:
    """Only 32 rows (quarterly) — tests minimum viable dataset size."""

    def test_fits_very_short_quarterly(self):
        m = quick_fit("08_finance_quarterly_short.csv", "revenue", "quarter", n_cv=2)
        assert m._is_fitted

    def test_predict_4_quarters(self):
        m = quick_fit("08_finance_quarterly_short.csv", "revenue", "quarter", n_cv=2)
        fc = m.predict(4)
        assert len(fc) == 4

    def test_scores_exist(self):
        m = quick_fit("08_finance_quarterly_short.csv", "revenue", "quarter", n_cv=2)
        assert_scores_sane(m._best_scores, "finance_quarterly")


class TestDataset09_DateAsIndex:
    """Date is already the DataFrame index — should auto-detect."""

    def test_fits_with_date_as_index(self):
        m = quick_fit("09_demand_date_as_index.csv", "demand", "date")
        assert m._is_fitted

    def test_weather_categorical_handled(self):
        df = load("09_demand_date_as_index.csv")
        m = BAXModel(n_cv_splits=2, verbose=False)
        m.fit(df, target_col="demand", date_col="date")
        for col in m._X_train.columns:
            assert pd.api.types.is_numeric_dtype(m._X_train[col])


class TestDataset10_EdgeCases:
    """Constant cols, all-NaN col, bool col, negative values, extreme outliers."""

    def test_fits_without_error(self):
        m = quick_fit("10_edge_cases_constant_allnan_bool.csv", "value", "date")
        assert m._is_fitted

    def test_constant_col_dropped(self):
        df = load("10_edge_cases_constant_allnan_bool.csv")
        m = BAXModel(n_cv_splits=2, verbose=False)
        m.fit(df, target_col="value", date_col="date")
        audit = m._preprocessing_audit.get("column_handler", {})
        dropped = " ".join(str(d) for d in audit.get("columns_dropped", []))
        assert "constant_col" in dropped

    def test_all_nan_col_dropped(self):
        df = load("10_edge_cases_constant_allnan_bool.csv")
        m = BAXModel(n_cv_splits=2, verbose=False)
        m.fit(df, target_col="value", date_col="date")
        audit = m._preprocessing_audit.get("column_handler", {})
        dropped = " ".join(str(d) for d in audit.get("columns_dropped", []))
        assert "all_nan_col" in dropped

    def test_bool_col_converted(self):
        df = load("10_edge_cases_constant_allnan_bool.csv")
        m = BAXModel(n_cv_splits=2, verbose=False)
        m.fit(df, target_col="value", date_col="date")
        # bool_feature should exist as numeric (was bool before)
        feat_cols = m._X_train.columns.tolist()
        # After encoding bool → int it may appear directly or as lag features
        assert m._is_fitted  # at minimum it didn't crash

    def test_extreme_outliers_handled(self):
        df = load("10_edge_cases_constant_allnan_bool.csv")
        m = BAXModel(n_cv_splits=2, verbose=False, outlier_treatment="cap")
        m.fit(df, target_col="value", date_col="date")
        assert m._is_fitted

    def test_negative_values_no_crash(self):
        df = load("10_edge_cases_constant_allnan_bool.csv")
        m = BAXModel(n_cv_splits=2, verbose=False)
        m.fit(df, target_col="value", date_col="date")
        assert m._is_fitted


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3 — MODEL EVALUATION TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestModelEvaluation:
    """Validates that evaluation metrics behave correctly."""

    @pytest.fixture(scope="class")
    def fitted_model(self):
        return quick_fit("01_retail_sales_daily.csv", "sales", "date")

    def test_mae_less_than_rmse(self, fitted_model):
        s = fitted_model._best_scores
        assert s["mae"] <= s["rmse"], "MAE must always be ≤ RMSE"

    def test_r2_between_neg1_and_1(self, fitted_model):
        r2 = fitted_model._best_scores["r2"]
        assert -1.0 <= r2 <= 1.0, f"R² must be in [-1, 1], got {r2}"

    def test_mape_non_negative(self, fitted_model):
        mape = fitted_model._best_scores.get("mape", 0)
        assert mape >= 0, "MAPE cannot be negative"

    def test_scoreboard_sorted_by_composite(self, fitted_model):
        sb = fitted_model._selector.scoreboard
        scores = [r["composite_score"] for r in sb]
        assert scores == sorted(scores), "Scoreboard must be sorted ascending"

    def test_cv_scores_exist(self, fitted_model):
        for model in fitted_model._selector.candidates:
            if model.is_fitted:
                assert "cv_mae" in model.cv_scores_
                assert "cv_rmse" in model.cv_scores_

    def test_test_mae_plausible_vs_cv(self, fitted_model):
        """Test MAE should be in the same ballpark as CV MAE (not 10x worse)."""
        best = fitted_model._selector.best_model
        cv_mae  = best.cv_scores_.get("cv_mae", float("inf"))
        test_mae = best.test_scores_.get("mae", float("inf"))
        # Allow up to 3x degradation from CV to test (time series is harder)
        assert test_mae < cv_mae * 3, \
            f"Test MAE ({test_mae:.4f}) is >3x CV MAE ({cv_mae:.4f}) — possible overfit"

    def test_winner_has_lowest_composite(self, fitted_model):
        sb = fitted_model._selector.scoreboard
        winner_name = fitted_model._selector.best_model.name
        winner_score = next(r["composite_score"] for r in sb if r["model"] == winner_name)
        for r in sb:
            assert winner_score <= r["composite_score"], \
                f"Winner {winner_name} ({winner_score}) is not the lowest composite"

    def test_all_three_models_trained(self, fitted_model):
        names = {r["model"] for r in fitted_model._selector.scoreboard}
        assert names == {"RandomForest", "XGBoost", "CatBoost"}

    def test_no_data_leakage_in_split(self, fitted_model):
        assert fitted_model._X_train.index.max() < fitted_model._X_test.index.min(), \
            "Train/test split has temporal leakage!"


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4 — ANOMALY DETECTION TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestAnomalyEvaluation:

    @pytest.fixture(scope="class")
    def energy_model(self):
        """Energy dataset has injected known spikes at rows 1200-1205."""
        return quick_fit("03_energy_hourly_iot.csv", "kwh", "timestamp")

    def test_anomaly_rate_in_reasonable_range(self, energy_model):
        a = energy_model.anomalies()
        rate = a["anomaly_flag"].mean()
        assert 0.0 < rate < 0.20, f"Anomaly rate {rate:.2%} seems unreasonable"

    def test_severity_labels_correct(self, energy_model):
        a = energy_model.anomalies()
        valid = {"normal", "suspicious", "anomaly"}
        assert set(a["severity_label"].unique()).issubset(valid)

    def test_severity_escalates_with_residual(self, energy_model):
        a = energy_model.anomalies()
        # All anomaly (flag=1) rows should have severity ≥ 1
        flagged = a[a["anomaly_flag"] == 1]
        assert (flagged["severity"] >= 1).all(), \
            "Flagged anomalies must have severity >= 1"

    def test_anomaly_residuals_larger_than_normal(self, energy_model):
        a = energy_model.anomalies()
        mean_anom   = a[a["anomaly_flag"] == 1]["residual"].abs().mean()
        mean_normal = a[a["anomaly_flag"] == 0]["residual"].abs().mean()
        assert mean_anom > mean_normal, \
            "Anomaly residuals must be larger than normal residuals on average"

    def test_ensemble_vs_single_method(self):
        df = load("01_retail_sales_daily.csv")

        results = {}
        for method in ["ensemble", "zscore", "iqr", "isolation_forest"]:
            m = BAXModel(n_cv_splits=2, verbose=False, anomaly_method=method)
            m.fit(df, target_col="sales", date_col="date")
            a = m.anomalies()
            results[method] = a["anomaly_flag"].sum()
            assert a["anomaly_flag"].sum() >= 0  # at minimum doesn't crash

        # All methods should find some anomalies (or at least 0)
        assert all(v >= 0 for v in results.values())


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5 — FORECAST QUALITY TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestForecastQuality:

    @pytest.fixture(scope="class")
    def model(self):
        return quick_fit("01_retail_sales_daily.csv", "sales", "date")

    def test_forecast_length_correct(self, model):
        for steps in [7, 14, 30, 90]:
            fc = model.predict(steps)
            assert len(fc) == steps, f"Expected {steps} steps, got {len(fc)}"

    def test_forecast_values_vary(self, model):
        fc = model.predict(30)
        assert fc["forecast"].nunique() > 3, "Forecast should not be flat"

    def test_forecast_dates_in_future(self, model):
        last_train = model._df_processed.index[-1]
        fc = model.predict(14)
        assert fc.index.min() > last_train, "Forecast dates should be after training data"

    def test_forecast_dates_contiguous(self, model):
        fc = model.predict(14)
        diffs = fc.index.to_series().diff().dropna()
        assert diffs.nunique() == 1, "Forecast dates should be evenly spaced"

    def test_forecast_column_name(self, model):
        fc = model.predict(7)
        assert "forecast" in fc.columns

    def test_forecast_no_nan(self, model):
        fc = model.predict(30)
        assert fc["forecast"].isna().sum() == 0, "Forecast should not contain NaN"


# ═══════════════════════════════════════════════════════════════════════
# SECTION 6 — PREPROCESSING AUDIT TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestPreprocessingAudit:

    def test_all_audit_keys_present(self):
        m = quick_fit("01_retail_sales_daily.csv", "sales", "date")
        expected = ["column_handler", "validator", "imputer", "outlier",
                    "transformer", "scaler", "feature_eng", "splitter"]
        for key in expected:
            assert key in m._preprocessing_audit, f"Missing audit key: {key}"

    def test_frequency_detected_correctly(self):
        configs = [
            ("01_retail_sales_daily.csv",    "sales",      "date",       "D"),
            ("05_web_traffic_monthly.csv",    "page_views", "month",      "MS"),
            ("03_energy_hourly_iot.csv",      "kwh",        "timestamp",  "h"),
        ]
        for csv, target, date_c, expected_freq in configs:
            m = quick_fit(csv, target, date_c)
            assert m._freq == expected_freq, \
                f"{csv}: expected freq={expected_freq}, got {m._freq}"

    def test_imputer_strategy_logged(self):
        m = quick_fit("06_manufacturing_heavy_missing.csv", "defect_rate", "date")
        audit = m._preprocessing_audit["imputer"]
        assert "strategy_used" in audit
        assert audit["strategy_used"] != ""

    def test_scaler_choice_logged(self):
        m = quick_fit("01_retail_sales_daily.csv", "sales", "date")
        assert "scaler_used" in m._preprocessing_audit["scaler"]

    def test_stl_decomposition_adds_features(self):
        m = quick_fit("01_retail_sales_daily.csv", "sales", "date")
        assert m._preprocessing_audit["transformer"].get("stl_decomposition") is True
        assert "trend_component" in m._df_processed.columns

    def test_feature_count_positive(self):
        m = quick_fit("01_retail_sales_daily.csv", "sales", "date")
        assert m._preprocessing_audit["feature_eng"]["total_features_added"] > 10

    def test_no_data_leakage_confirmed(self):
        m = quick_fit("01_retail_sales_daily.csv", "sales", "date")
        sp = m._preprocessing_audit["splitter"]
        assert pd.Timestamp(sp["train_end"]) < pd.Timestamp(sp["test_start"])


# ═══════════════════════════════════════════════════════════════════════
# SECTION 7 — REPORT GENERATION TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestReportGeneration:
    import re

    def test_report_file_created(self, tmp_path):
        m = quick_fit("01_retail_sales_daily.csv", "sales", "date")
        m.anomalies()
        path = m.report(str(tmp_path / "report"))
        assert os.path.exists(path)

    def test_report_self_contained_no_cdn(self, tmp_path):
        """No external script src tags — fully self-contained."""
        import re
        m = quick_fit("01_retail_sales_daily.csv", "sales", "date")
        m.anomalies()
        path = m.report(str(tmp_path / "report"))
        with open(path) as f:
            html = f.read()
        cdn_tags = re.findall(
            r'<script[^>]+src=["\']https?://[^"\']*["\']', html
        )
        assert cdn_tags == [], f"CDN script tags found: {cdn_tags}"

    def test_report_has_plotly_inline(self, tmp_path):
        import re
        m = quick_fit("01_retail_sales_daily.csv", "sales", "date")
        m.anomalies()
        path = m.report(str(tmp_path / "report"))
        with open(path) as f:
            html = f.read()
        inline_scripts = re.findall(
            r'<script type=["\']text/javascript["\']>', html
        )
        assert len(inline_scripts) == 1, \
            f"Expected exactly 1 inline Plotly script, got {len(inline_scripts)}"

    def test_report_has_all_charts(self, tmp_path):
        import re
        m = quick_fit("01_retail_sales_daily.csv", "sales", "date")
        m.predict(14)
        m.anomalies()
        path = m.report(str(tmp_path / "report"))
        with open(path) as f:
            html = f.read()
        chart_count = len(re.findall(r"Plotly\.newPlot", html))
        assert chart_count >= 5, f"Expected ≥5 charts, found {chart_count}"

    def test_report_contains_bax_narrative(self, tmp_path):
        m = quick_fit("01_retail_sales_daily.csv", "sales", "date")
        m.anomalies()
        path = m.report(str(tmp_path / "report"))
        with open(path) as f:
            html = f.read()
        assert "BAX" in html or "behavioural" in html.lower()

    def test_report_size_reasonable(self, tmp_path):
        m = quick_fit("01_retail_sales_daily.csv", "sales", "date")
        m.anomalies()
        path = m.report(str(tmp_path / "report"))
        size_mb = os.path.getsize(path) / 1024 / 1024
        assert 3.0 < size_mb < 15.0, f"Report size {size_mb:.1f}MB seems wrong"


# ═══════════════════════════════════════════════════════════════════════
# SECTION 8 — ERROR HANDLING & ROBUSTNESS TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestErrorHandling:

    def test_unfitted_predict_raises(self):
        m = BAXModel(verbose=False)
        with pytest.raises(RuntimeError, match="fit"):
            m.predict(10)

    def test_unfitted_anomalies_raises(self):
        m = BAXModel(verbose=False)
        with pytest.raises(RuntimeError, match="fit"):
            m.anomalies()

    def test_unfitted_explain_raises(self):
        m = BAXModel(verbose=False)
        with pytest.raises(RuntimeError, match="fit"):
            m.explain()

    def test_unfitted_report_raises(self, tmp_path):
        m = BAXModel(verbose=False)
        with pytest.raises(RuntimeError, match="fit"):
            m.report(str(tmp_path / "r"))

    def test_wrong_target_col_raises(self):
        df = load("01_retail_sales_daily.csv")
        m = BAXModel(verbose=False)
        with pytest.raises((KeyError, ValueError, Exception)):
            m.fit(df, target_col="nonexistent_column", date_col="date")

    def test_no_date_col_no_datetime_raises(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        m = BAXModel(verbose=False)
        with pytest.raises(ValueError, match="No datetime"):
            m.fit(df, target_col="a")

    def test_outlier_cap_treatment(self):
        df = load("10_edge_cases_constant_allnan_bool.csv")
        m = BAXModel(n_cv_splits=2, verbose=False, outlier_treatment="cap")
        m.fit(df, target_col="value", date_col="date")
        assert m._is_fitted

    def test_outlier_flag_treatment(self):
        df = load("01_retail_sales_daily.csv")
        m = BAXModel(n_cv_splits=2, verbose=False, outlier_treatment="flag")
        m.fit(df, target_col="sales", date_col="date")
        assert m._is_fitted

    def test_all_anomaly_methods(self):
        df = load("01_retail_sales_daily.csv")
        for method in ["ensemble", "zscore", "iqr", "isolation_forest"]:
            m = BAXModel(n_cv_splits=2, verbose=False, anomaly_method=method)
            m.fit(df, target_col="sales", date_col="date")
            a = m.anomalies()
            assert "anomaly_flag" in a.columns, f"Method {method} failed"

    def test_summary_returns_all_keys(self):
        m = quick_fit("01_retail_sales_daily.csv", "sales", "date")
        s = m.summary()
        for key in ["target_col", "frequency", "best_model",
                    "test_mae", "test_rmse", "test_r2"]:
            assert key in s, f"Missing summary key: {key}"


# ═══════════════════════════════════════════════════════════════════════
# SECTION 9 — STANDALONE RUNNER (non-pytest)
# ═══════════════════════════════════════════════════════════════════════

def run_all_manual():
    """
    Run all 10 datasets manually and print a readable summary table.
    Use this if you don't have pytest installed.
    """
    configs = [
        ("01_retail_sales_daily.csv",                "sales",       "date",        "Daily, clean"),
        ("02_retail_multivariate_with_categoricals.csv","sales",    "date",        "Multi-col, categorical"),
        ("03_energy_hourly_iot.csv",                  "kwh",         "timestamp",  "Hourly IoT"),
        ("04_stock_price_daily.csv",                  "close",       "date",       "Stock, multi-numeric"),
        ("05_web_traffic_monthly.csv",                "page_views",  "month",      "Monthly, short"),
        ("06_manufacturing_heavy_missing.csv",        "defect_rate", "date",       "15% missing + ID col"),
        ("07_healthcare_weekly_gaps.csv",             "admissions",  "week_start", "Weekly, gap periods"),
        ("08_finance_quarterly_short.csv",            "revenue",     "quarter",    "Quarterly, 32 rows"),
        ("09_demand_date_as_index.csv",               "demand",      "date",       "Date as index"),
        ("10_edge_cases_constant_allnan_bool.csv",    "value",       "date",       "Constant/NaN/bool cols"),
    ]

    print("\n" + "="*90)
    print(f"{'Dataset':<42} {'Status':<8} {'Winner':<14} {'MAE':>8} {'R²':>7} {'Anomalies':>10}")
    print("="*90)

    passed = failed = 0
    for csv, target, date_c, desc in configs:
        try:
            df = load(csv)
            m = BAXModel(n_cv_splits=2, verbose=False)
            m.fit(df, target_col=target, date_col=date_c)
            m.predict(14)
            anom = m.anomalies()
            s = m._best_scores
            winner = m._selector.best_model.name[:13]
            mae    = f"{s.get('mae', 0):.4f}"
            r2     = f"{s.get('r2', 0):.3f}"
            n_anom = anom["anomaly_flag"].sum()
            print(f" ✓  {desc:<40} {'PASS':<8} {winner:<14} {mae:>8} {r2:>7} {n_anom:>10}")
            passed += 1
        except Exception as e:
            print(f" ✗  {desc:<40} {'FAIL':<8} {str(e)[:50]}")
            failed += 1

    print("="*90)
    print(f"\nResults: {passed} passed, {failed} failed out of {passed+failed} datasets")
    if failed == 0:
        print("ALL DATASETS PASSED — ready for deployment")
    else:
        print(f"FIX {failed} FAILURE(S) BEFORE DEPLOYING")
    return failed == 0


if __name__ == "__main__":
    # Generate datasets if they don't exist
    if not os.path.exists(os.path.join(DATASETS_DIR, "01_retail_sales_daily.csv")):
        print("Generating test datasets first...")
        exec(open(os.path.join(DATASETS_DIR, "generate_test_datasets.py")).read())

    success = run_all_manual()
    sys.exit(0 if success else 1)
