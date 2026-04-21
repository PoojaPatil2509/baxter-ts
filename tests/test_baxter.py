"""
baxter-ts test suite.
Run with:  pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ── Fixtures ─────────────────────────────────────────────────────────

def make_daily_df(n=365, noise=0.1, missing_pct=0.03, seed=42):
    """Synthetic daily sales data with trend, seasonality, and noise."""
    np.random.seed(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    trend = np.linspace(100, 180, n)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n) / 7)
    noise_arr = np.random.randn(n) * noise * 100
    values = trend + seasonal + noise_arr
    df = pd.DataFrame({"date": dates, "sales": values})
    # Inject missing values
    miss_idx = np.random.choice(n, size=int(n * missing_pct), replace=False)
    df.loc[miss_idx, "sales"] = np.nan
    return df


def make_hourly_df(n=500, seed=7):
    np.random.seed(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    values = (
        50
        + 15 * np.sin(2 * np.pi * np.arange(n) / 24)
        + np.random.randn(n) * 3
    )
    return pd.DataFrame({"timestamp": dates, "kwh": values})


def make_monthly_df(n=60, seed=3):
    np.random.seed(seed)
    dates = pd.date_range("2019-01-01", periods=n, freq="MS")
    values = 1000 + np.cumsum(np.random.randn(n) * 50) + 10 * np.arange(n)
    return pd.DataFrame({"date": dates, "revenue": values})


# ── Preprocessing tests ───────────────────────────────────────────────

class TestDatetimeValidator:
    def test_auto_detect_date_col(self):
        from baxter_ts.preprocessing.validator import DatetimeValidator
        df = make_daily_df()
        v = DatetimeValidator()
        out = v.fit_transform(df, target_col="sales")
        assert isinstance(out.index, pd.DatetimeIndex)

    def test_explicit_date_col(self):
        from baxter_ts.preprocessing.validator import DatetimeValidator
        df = make_daily_df()
        v = DatetimeValidator()
        out = v.fit_transform(df, date_col="date")
        assert isinstance(out.index, pd.DatetimeIndex)

    def test_frequency_inference(self):
        from baxter_ts.preprocessing.validator import DatetimeValidator
        df = make_daily_df()
        v = DatetimeValidator()
        v.fit_transform(df)
        assert v.detected_freq is not None

    def test_sorted_index(self):
        from baxter_ts.preprocessing.validator import DatetimeValidator
        df = make_daily_df()
        df = df.sample(frac=1, random_state=1)  # shuffle
        v = DatetimeValidator()
        out = v.fit_transform(df)
        assert out.index.is_monotonic_increasing

    def test_duplicate_removal(self):
        from baxter_ts.preprocessing.validator import DatetimeValidator
        df = make_daily_df(n=100)
        df = pd.concat([df, df.iloc[:5]])  # add duplicates
        v = DatetimeValidator()
        out = v.fit_transform(df)
        assert not out.index.duplicated().any()

    def test_hourly_frequency(self):
        from baxter_ts.preprocessing.validator import DatetimeValidator
        df = make_hourly_df()
        v = DatetimeValidator()
        out = v.fit_transform(df, date_col="timestamp")
        assert isinstance(out.index, pd.DatetimeIndex)


class TestImputer:
    def test_no_missing(self):
        from baxter_ts.preprocessing.imputer import TimeSeriesImputer
        df = make_daily_df(missing_pct=0)
        df = df.set_index("date")
        imp = TimeSeriesImputer()
        out = imp.fit_transform(df, "sales")
        assert out["sales"].isna().sum() == 0
        assert imp.audit["strategy_used"] == "none_needed"

    def test_fills_missing(self):
        from baxter_ts.preprocessing.imputer import TimeSeriesImputer
        df = make_daily_df(missing_pct=0.05)
        df = df.set_index("date")
        imp = TimeSeriesImputer()
        out = imp.fit_transform(df, "sales")
        assert out["sales"].isna().sum() == 0

    def test_strategy_selection(self):
        from baxter_ts.preprocessing.imputer import TimeSeriesImputer
        df = make_daily_df(missing_pct=0.01, n=200)
        df = df.set_index("date")
        imp = TimeSeriesImputer(strategy="auto")
        imp.fit_transform(df, "sales")
        assert imp._used_strategy == "ffill"


class TestOutlierHandler:
    def test_cap_treatment(self):
        from baxter_ts.preprocessing.outlier import OutlierHandler
        df = make_daily_df()
        df = df.set_index("date")
        df.loc[df.index[10], "sales"] = 99999  # inject outlier
        handler = OutlierHandler(treatment="cap")
        out = handler.fit_transform(df, "sales")
        assert out["sales"].max() < 99999

    def test_flag_treatment(self):
        from baxter_ts.preprocessing.outlier import OutlierHandler
        df = make_daily_df()
        df = df.set_index("date")
        handler = OutlierHandler(treatment="flag")
        out = handler.fit_transform(df, "sales")
        assert "sales_is_outlier" in out.columns

    def test_audit_populated(self):
        from baxter_ts.preprocessing.outlier import OutlierHandler
        df = make_daily_df()
        df = df.set_index("date")
        handler = OutlierHandler()
        handler.fit_transform(df, "sales")
        assert "outliers_found" in handler.audit


class TestTransformer:
    def test_returns_dataframe(self):
        from baxter_ts.preprocessing.transformer import StationarityTransformer
        df = make_daily_df(missing_pct=0)
        df = df.set_index("date")
        t = StationarityTransformer()
        out = t.fit_transform(df, "sales", freq="D")
        assert isinstance(out, pd.DataFrame)

    def test_audit_keys(self):
        from baxter_ts.preprocessing.transformer import StationarityTransformer
        df = make_daily_df(missing_pct=0)
        df = df.set_index("date")
        t = StationarityTransformer()
        t.fit_transform(df, "sales", freq="D")
        assert "adf_pvalue" in t.audit
        assert "diffs_applied" in t.audit


class TestFeatureEngineer:
    def test_adds_lag_features(self):
        from baxter_ts.preprocessing.feature_eng import TimeSeriesFeatureEngineer
        df = make_daily_df(missing_pct=0)
        df = df.set_index("date")
        fe = TimeSeriesFeatureEngineer()
        out = fe.fit_transform(df, "sales", freq="D")
        assert any(c.startswith("lag_") for c in out.columns)

    def test_adds_rolling_features(self):
        from baxter_ts.preprocessing.feature_eng import TimeSeriesFeatureEngineer
        df = make_daily_df(missing_pct=0)
        df = df.set_index("date")
        fe = TimeSeriesFeatureEngineer()
        out = fe.fit_transform(df, "sales", freq="D")
        assert any(c.startswith("roll_mean_") for c in out.columns)

    def test_adds_calendar_features(self):
        from baxter_ts.preprocessing.feature_eng import TimeSeriesFeatureEngineer
        df = make_daily_df(missing_pct=0)
        df = df.set_index("date")
        fe = TimeSeriesFeatureEngineer(add_calendar=True)
        out = fe.fit_transform(df, "sales", freq="D")
        assert "dayofweek" in out.columns
        assert "month" in out.columns

    def test_adds_fourier_features(self):
        from baxter_ts.preprocessing.feature_eng import TimeSeriesFeatureEngineer
        df = make_daily_df(missing_pct=0)
        df = df.set_index("date")
        fe = TimeSeriesFeatureEngineer(add_fourier=True, fourier_order=2)
        out = fe.fit_transform(df, "sales", freq="D")
        assert "sin_k1" in out.columns
        assert "cos_k1" in out.columns

    def test_hourly_lags(self):
        from baxter_ts.preprocessing.feature_eng import TimeSeriesFeatureEngineer
        df = make_hourly_df()
        df = df.set_index("timestamp")
        fe = TimeSeriesFeatureEngineer()
        out = fe.fit_transform(df, "kwh", freq="h")
        assert any(c.startswith("lag_") for c in out.columns)


class TestSplitter:
    def test_no_leakage(self):
        from baxter_ts.preprocessing.splitter import TemporalSplitter
        df = make_daily_df(missing_pct=0)
        df = df.set_index("date")
        sp = TemporalSplitter(test_size=0.2)
        X_tr, X_te, y_tr, y_te = sp.split(df, "sales")
        assert X_tr.index.max() < X_te.index.min()

    def test_sizes(self):
        from baxter_ts.preprocessing.splitter import TemporalSplitter
        df = make_daily_df(n=200, missing_pct=0)
        df = df.set_index("date")
        sp = TemporalSplitter(test_size=0.2)
        X_tr, X_te, y_tr, y_te = sp.split(df, "sales")
        assert len(X_tr) + len(X_te) == 200


# ── Model tests ──────────────────────────────────────────────────────

class TestModels:
    def _prep(self):
        from baxter_ts.preprocessing.feature_eng import TimeSeriesFeatureEngineer
        from baxter_ts.preprocessing.splitter import TemporalSplitter
        df = make_daily_df(n=300, missing_pct=0)
        df = df.set_index("date")
        fe = TimeSeriesFeatureEngineer()
        df = fe.fit_transform(df, "sales", freq="D")
        df = df.fillna(0)
        sp = TemporalSplitter(test_size=0.2)
        return sp.split(df, "sales")

    def test_rf_fits_and_scores(self):
        from baxter_ts.models.rf_model import RFModel
        X_tr, X_te, y_tr, y_te = self._prep()
        m = RFModel(n_estimators=10)
        m.fit(X_tr, y_tr)
        scores = m.score(X_te, y_te)
        assert "mae" in scores and scores["mae"] >= 0

    def test_xgb_fits_and_scores(self):
        from baxter_ts.models.xgb_model import XGBModel
        X_tr, X_te, y_tr, y_te = self._prep()
        m = XGBModel(n_estimators=20)
        m.fit(X_tr, y_tr)
        scores = m.score(X_te, y_te)
        assert "rmse" in scores and scores["rmse"] >= 0

    def test_catboost_fits_and_scores(self):
        from baxter_ts.models.catboost_model import CatModel
        X_tr, X_te, y_tr, y_te = self._prep()
        m = CatModel(iterations=20)
        m.fit(X_tr, y_tr)
        scores = m.score(X_te, y_te)
        assert "r2" in scores

    def test_selector_picks_winner(self):
        from baxter_ts.models.selector import ModelSelector
        X_tr, X_te, y_tr, y_te = self._prep()
        sel = ModelSelector(n_cv_splits=2)
        sel.fit(X_tr, y_tr, X_te, y_te)
        assert sel.best_model is not None
        assert sel.best_model.name in ["RandomForest", "XGBoost", "CatBoost"]


# ── Anomaly tests ────────────────────────────────────────────────────

class TestAnomalyDetector:
    def test_returns_dataframe(self):
        from baxter_ts.anomaly.detector import AnomalyDetector
        y_true = pd.Series(np.random.randn(100) + 10,
                           index=pd.date_range("2023-01-01", periods=100))
        y_pred = y_true.values + np.random.randn(100) * 0.5
        det = AnomalyDetector()
        out = det.fit_predict(y_true, y_pred)
        assert "anomaly_flag" in out.columns
        assert "severity_label" in out.columns

    def test_anomaly_flag_binary(self):
        from baxter_ts.anomaly.detector import AnomalyDetector
        y_true = pd.Series(np.random.randn(100) + 50,
                           index=pd.date_range("2023-01-01", periods=100))
        y_pred = y_true.values + np.random.randn(100)
        det = AnomalyDetector()
        out = det.fit_predict(y_true, y_pred)
        assert set(out["anomaly_flag"].unique()).issubset({0, 1})


# ── End-to-end BAXModel tests ─────────────────────────────────────────

class TestBAXModel:
    def test_fit_daily(self):
        from baxter_ts import BAXModel
        df = make_daily_df(n=400)
        m = BAXModel(verbose=False)
        m.fit(df, target_col="sales", date_col="date")
        assert m._is_fitted

    def test_fit_hourly(self):
        from baxter_ts import BAXModel
        df = make_hourly_df(n=600)
        m = BAXModel(verbose=False)
        m.fit(df, target_col="kwh", date_col="timestamp")
        assert m._is_fitted

    def test_fit_monthly(self):
        from baxter_ts import BAXModel
        df = make_monthly_df(n=72)
        m = BAXModel(verbose=False)
        m.fit(df, target_col="revenue", date_col="date")
        assert m._is_fitted

    def test_predict_returns_dataframe(self):
        from baxter_ts import BAXModel
        df = make_daily_df(n=300)
        m = BAXModel(verbose=False)
        m.fit(df, target_col="sales", date_col="date")
        forecast = m.predict(steps=14)
        assert isinstance(forecast, pd.DataFrame)
        assert len(forecast) == 14
        assert "forecast" in forecast.columns

    def test_anomalies_returns_dataframe(self):
        from baxter_ts import BAXModel
        df = make_daily_df(n=300)
        m = BAXModel(verbose=False)
        m.fit(df, target_col="sales", date_col="date")
        anom = m.anomalies()
        assert isinstance(anom, pd.DataFrame)
        assert "anomaly_flag" in anom.columns

    def test_explain_returns_string(self):
        from baxter_ts import BAXModel
        df = make_daily_df(n=300)
        m = BAXModel(verbose=False)
        m.fit(df, target_col="sales", date_col="date")
        narrative = m.explain()
        assert isinstance(narrative, str)
        assert len(narrative) > 50

    def test_summary_keys(self):
        from baxter_ts import BAXModel
        df = make_daily_df(n=300)
        m = BAXModel(verbose=False)
        m.fit(df, target_col="sales", date_col="date")
        s = m.summary()
        for key in ["best_model", "test_mae", "test_rmse", "frequency"]:
            assert key in s

    def test_scoreboard_dataframe(self):
        from baxter_ts import BAXModel
        df = make_daily_df(n=300)
        m = BAXModel(verbose=False)
        m.fit(df, target_col="sales", date_col="date")
        sb = m.scoreboard()
        assert isinstance(sb, pd.DataFrame)
        assert len(sb) == 3

    def test_report_creates_file(self, tmp_path):
        from baxter_ts import BAXModel
        df = make_daily_df(n=300)
        m = BAXModel(verbose=False)
        m.fit(df, target_col="sales", date_col="date")
        m.anomalies()
        path = m.report(str(tmp_path / "test_report"))
        assert path.endswith(".html")
        import os
        assert os.path.exists(path)

    def test_unfitted_raises(self):
        from baxter_ts import BAXModel
        m = BAXModel(verbose=False)
        with pytest.raises(RuntimeError):
            m.predict(10)
        with pytest.raises(RuntimeError):
            m.anomalies()
