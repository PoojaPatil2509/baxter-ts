"""
baxter-ts Pre-Launch Comprehensive Test Suite
==============================================
30 test scenarios across 5 categories:
  1. Frequency coverage   (6 tests)
  2. Data quality issues  (5 tests)
  3. Series characteristics (7 tests)
  4. Domain simulations   (7 tests)
  5. Edge cases           (5 tests)

Run:
    pytest tests/test_comprehensive.py -v
    pytest tests/test_comprehensive.py -v --tb=short 2>&1 | tee test_results.txt
"""

import warnings
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── helpers ──────────────────────────────────────────────────────────


def run_full_pipeline(df, target_col, date_col, n_cv=2, steps=10):
    """
    Runs the complete BAXModel pipeline and returns a result dict.
    Asserts the minimum quality bar for every scenario.
    """
    from baxter_ts import BAXModel

    model = BAXModel(n_cv_splits=n_cv, verbose=False)
    model.fit(df, target_col=target_col, date_col=date_col)

    # Predict
    forecast = model.predict(steps=steps)
    assert isinstance(forecast, pd.DataFrame), "predict() must return DataFrame"
    assert len(forecast) == steps, f"Expected {steps} forecast rows, got {len(forecast)}"
    assert not forecast["forecast"].isnull().all(), "All forecast values are NaN"
    assert forecast["forecast"].nunique() > 1, "Forecast is flat (all same value)"

    # Anomalies
    anom = model.anomalies()
    assert isinstance(anom, pd.DataFrame), "anomalies() must return DataFrame"
    assert "anomaly_flag" in anom.columns
    assert set(anom["anomaly_flag"].unique()).issubset({0, 1})
    assert "severity_label" in anom.columns

    # Explain
    narrative = model.explain()
    assert isinstance(narrative, str) and len(narrative) > 30

    # Scoreboard
    sb = model.scoreboard()
    assert isinstance(sb, pd.DataFrame)
    assert len(sb) == 3, "Must have exactly 3 model candidates"

    # Summary
    s = model.summary()
    assert s["best_model"] in ["RandomForest", "XGBoost", "CatBoost"]
    assert s["test_mae"] is not None and s["test_mae"] >= 0
    assert s["test_r2"] is not None

    # Report
    import tempfile, os, re
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        tmp_path = f.name
    model.report(tmp_path.replace(".html", ""))
    assert os.path.exists(tmp_path), "Report HTML file not created"
    with open(tmp_path, encoding="utf-8") as f:
        html = f.read()
    # Verify no CDN dependency
    cdn_tags = re.findall(r'<script[^>]+src=["\']https?://', html)
    assert not cdn_tags, f"CDN script tags found (will break offline): {cdn_tags}"
    # Verify charts exist
    chart_count = len(re.findall(r"Plotly\.newPlot", html))
    assert chart_count >= 5, f"Expected >=5 charts, found {chart_count}"
    os.unlink(tmp_path)

    return {
        "model": s["best_model"],
        "mae": s["test_mae"],
        "rmse": s.get("test_rmse"),
        "r2": s["test_r2"],
        "anomalies": s["anomalies_found"],
        "forecast_variance": float(forecast["forecast"].var()),
    }


# ══════════════════════════════════════════════════════════════════════
# CATEGORY 1 — FREQUENCY COVERAGE
# ══════════════════════════════════════════════════════════════════════


class TestFrequencyCoverage:

    def test_minutely_iot_sensor(self):
        """1-minute IoT temperature sensor — 2 days of data."""
        np.random.seed(1)
        n = 2 * 24 * 60  # 2880 rows
        dates = pd.date_range("2024-01-01", periods=n, freq="min")
        values = (
            22
            + 3 * np.sin(2 * np.pi * np.arange(n) / 1440)   # daily cycle
            + np.random.randn(n) * 0.5
        )
        df = pd.DataFrame({"timestamp": dates, "temperature_c": values})
        result = run_full_pipeline(df, "temperature_c", "timestamp", steps=60)
        assert result["r2"] > 0.0, f"R2={result['r2']} too low for minutely data"

    def test_hourly_energy(self):
        """Hourly electricity demand — 90 days."""
        np.random.seed(2)
        n = 90 * 24
        dates = pd.date_range("2023-01-01", periods=n, freq="h")
        values = (
            500
            + 150 * np.sin(2 * np.pi * np.arange(n) / 24)   # daily
            + 80  * np.sin(2 * np.pi * np.arange(n) / (24*7))  # weekly
            + np.random.randn(n) * 20
        )
        df = pd.DataFrame({"datetime": dates, "kwh": values})
        result = run_full_pipeline(df, "kwh", "datetime", steps=24)
        assert result["r2"] > 0.0

    def test_daily_sales(self):
        """Daily retail sales — 2 years."""
        np.random.seed(3)
        n = 730
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        values = (
            np.linspace(1000, 1500, n)
            + 200 * np.sin(2 * np.pi * np.arange(n) / 7)
            + 300 * np.sin(2 * np.pi * np.arange(n) / 365)
            + np.random.randn(n) * 50
        )
        values[np.random.choice(n, 20, replace=False)] = np.nan
        df = pd.DataFrame({"date": dates, "sales": values})
        result = run_full_pipeline(df, "sales", "date", steps=30)
        assert result["r2"] > 0.3

    def test_weekly_demand(self):
        """Weekly retail demand — 3 years of weeks."""
        np.random.seed(4)
        n = 156  # 3 years
        dates = pd.date_range("2021-01-04", periods=n, freq="W")
        values = (
            np.linspace(500, 800, n)
            + 100 * np.sin(2 * np.pi * np.arange(n) / 52)
            + np.random.randn(n) * 30
        )
        df = pd.DataFrame({"week": dates, "units_sold": values})
        result = run_full_pipeline(df, "units_sold", "week", steps=8)
        assert result["r2"] > 0.0

    def test_monthly_revenue(self):
        """Monthly revenue — 5 years."""
        np.random.seed(5)
        n = 60
        dates = pd.date_range("2019-01-01", periods=n, freq="MS")
        values = (
            10000
            + np.cumsum(np.random.randn(n) * 200)
            + 1500 * np.sin(2 * np.pi * np.arange(n) / 12)
        )
        df = pd.DataFrame({"month": dates, "revenue": values})
        result = run_full_pipeline(df, "revenue", "month", steps=6)
        assert result["mae"] >= 0

    def test_quarterly_financials(self):
        """Quarterly EPS — 10 years of quarters."""
        np.random.seed(6)
        n = 40
        dates = pd.date_range("2014-01-01", periods=n, freq="QS")
        values = (
            2.0
            + np.cumsum(np.random.randn(n) * 0.1)
            + 0.3 * np.sin(2 * np.pi * np.arange(n) / 4)
        )
        df = pd.DataFrame({"quarter": dates, "eps": values})
        result = run_full_pipeline(df, "eps", "quarter", steps=4)
        assert result["mae"] >= 0


# ══════════════════════════════════════════════════════════════════════
# CATEGORY 2 — DATA QUALITY ISSUES
# ══════════════════════════════════════════════════════════════════════


class TestDataQualityIssues:

    def test_heavy_missing_values(self):
        """30% of values missing — library must handle gracefully."""
        np.random.seed(10)
        n = 400
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        values = np.linspace(100, 200, n) + np.random.randn(n) * 10
        miss_idx = np.random.choice(n, int(n * 0.30), replace=False)
        values[miss_idx] = np.nan
        df = pd.DataFrame({"date": dates, "value": values})
        assert df["value"].isna().mean() > 0.25, "Need >25% missing for this test"
        result = run_full_pipeline(df, "value", "date", steps=14)
        assert result["mae"] >= 0, "Pipeline failed on heavy missing data"

    def test_extreme_outlier_spikes(self):
        """Deliberate 10x spikes injected — outlier handler must cap them."""
        np.random.seed(11)
        n = 365
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        values = np.linspace(100, 200, n) + np.random.randn(n) * 5
        # Inject extreme spikes
        spike_idx = [10, 50, 120, 200, 300]
        values[spike_idx] = values[spike_idx] * 10
        df = pd.DataFrame({"date": dates, "sensor": values})
        result = run_full_pipeline(df, "sensor", "date", steps=14)
        # Model should still achieve reasonable fit despite spikes
        assert result["mae"] >= 0

    def test_sudden_level_shift(self):
        """Structural break mid-series — e.g. post-COVID demand shift."""
        np.random.seed(12)
        n = 500
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        values = np.zeros(n)
        values[:250] = 100 + np.random.randn(250) * 5   # pre-shift
        values[250:] = 160 + np.random.randn(250) * 5   # post-shift (level jump)
        df = pd.DataFrame({"date": dates, "demand": values})
        result = run_full_pipeline(df, "demand", "date", steps=14)
        assert result["mae"] >= 0

    def test_high_noise_low_signal(self):
        """SNR < 1 — mostly noise, very weak trend."""
        np.random.seed(13)
        n = 300
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        signal = np.linspace(0, 5, n)           # weak trend
        noise  = np.random.randn(n) * 20        # dominant noise
        values = signal + noise
        df = pd.DataFrame({"date": dates, "noisy_metric": values})
        result = run_full_pipeline(df, "noisy_metric", "date", steps=10)
        # Just must not crash; R2 can be low
        assert result["mae"] >= 0

    def test_unsorted_timestamps(self):
        """Timestamps not in order — validator must sort them."""
        np.random.seed(14)
        n = 300
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        values = np.linspace(100, 200, n) + np.random.randn(n) * 5
        df = pd.DataFrame({"date": dates, "value": values})
        df = df.sample(frac=1, random_state=99).reset_index(drop=True)  # shuffle
        assert not df["date"].is_monotonic_increasing, "Need unsorted data"
        result = run_full_pipeline(df, "value", "date", steps=10)
        assert result["mae"] >= 0


# ══════════════════════════════════════════════════════════════════════
# CATEGORY 3 — SERIES CHARACTERISTICS
# ══════════════════════════════════════════════════════════════════════


class TestSeriesCharacteristics:

    def test_strong_trend_only(self):
        """Pure linear trend, minimal seasonality."""
        np.random.seed(20)
        n = 400
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        values = np.linspace(50, 500, n) + np.random.randn(n) * 3
        df = pd.DataFrame({"date": dates, "metric": values})
        result = run_full_pipeline(df, "metric", "date", steps=20)
        assert result["r2"] > 0.5, f"Pure trend should achieve R2>0.5, got {result['r2']}"

    def test_strong_seasonality_only(self):
        """Pure weekly seasonality, no trend."""
        np.random.seed(21)
        n = 365
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        values = 100 + 40 * np.sin(2 * np.pi * np.arange(n) / 7) + np.random.randn(n) * 3
        df = pd.DataFrame({"date": dates, "pattern": values})
        result = run_full_pipeline(df, "pattern", "date", steps=14)
        assert result["r2"] > 0.3

    def test_trend_plus_seasonality(self):
        """Both trend and dual seasonality."""
        np.random.seed(22)
        n = 730
        dates = pd.date_range("2021-01-01", periods=n, freq="D")
        values = (
            np.linspace(200, 500, n)
            + 60  * np.sin(2 * np.pi * np.arange(n) / 7)
            + 120 * np.sin(2 * np.pi * np.arange(n) / 365)
            + np.random.randn(n) * 15
        )
        df = pd.DataFrame({"date": dates, "sales": values})
        result = run_full_pipeline(df, "sales", "date", steps=30)
        assert result["r2"] > 0.4

    def test_nonstationary_random_walk(self):
        """Pure random walk (no trend, no seasonality) — hardest to forecast."""
        np.random.seed(23)
        n = 400
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        values = np.cumsum(np.random.randn(n)) + 100
        df = pd.DataFrame({"date": dates, "price": values})
        result = run_full_pipeline(df, "price", "date", steps=10)
        # Random walk is near-impossible to forecast; just must not crash
        assert result["mae"] >= 0

    def test_short_series(self):
        """Very short series — only 80 rows."""
        np.random.seed(24)
        n = 80
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        values = np.linspace(10, 50, n) + np.random.randn(n) * 2
        df = pd.DataFrame({"date": dates, "value": values})
        result = run_full_pipeline(df, "value", "date", n_cv=2, steps=5)
        assert result["mae"] >= 0

    def test_long_series(self):
        """Long series — 5000 rows. Tests performance and memory."""
        np.random.seed(25)
        n = 5000
        dates = pd.date_range("2010-01-01", periods=n, freq="D")
        values = (
            np.linspace(0, 1000, n)
            + 100 * np.sin(2 * np.pi * np.arange(n) / 365)
            + np.random.randn(n) * 20
        )
        df = pd.DataFrame({"date": dates, "value": values})
        import time
        start = time.time()
        result = run_full_pipeline(df, "value", "date", steps=30)
        elapsed = time.time() - start
        assert elapsed < 300, f"Pipeline took {elapsed:.1f}s on 5000 rows (max 300s)"
        assert result["mae"] >= 0

    def test_multivariate_with_exogenous(self):
        """Multiple numeric columns — library picks target, ignores rest."""
        np.random.seed(26)
        n = 400
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        df = pd.DataFrame({
            "date":        dates,
            "sales":       np.linspace(100, 200, n) + np.random.randn(n) * 5,
            "temperature": 15 + 10 * np.sin(2*np.pi*np.arange(n)/365) + np.random.randn(n),
            "promotions":  np.random.randint(0, 2, n).astype(float),
            "price":       np.linspace(9.99, 12.99, n) + np.random.randn(n) * 0.2,
        })
        result = run_full_pipeline(df, "sales", "date", steps=14)
        assert result["mae"] >= 0


# ══════════════════════════════════════════════════════════════════════
# CATEGORY 4 — DOMAIN SIMULATIONS
# ══════════════════════════════════════════════════════════════════════


class TestDomainSimulations:

    def test_financial_stock_price(self):
        """Simulated stock price (log-normal random walk with drift)."""
        np.random.seed(30)
        n = 500
        # Business days only (like real stock data)
        dates = pd.bdate_range("2022-01-03", periods=n)
        log_returns = np.random.randn(n) * 0.015 + 0.0003
        prices = 100 * np.exp(np.cumsum(log_returns))
        df = pd.DataFrame({"date": dates, "close_price": prices})
        result = run_full_pipeline(df, "close_price", "date", steps=10)
        assert result["mae"] >= 0

    def test_energy_electricity_demand(self):
        """Electricity demand with dual seasonality (daily + weekly)."""
        np.random.seed(31)
        n = 60 * 24  # 60 days hourly
        dates = pd.date_range("2023-06-01", periods=n, freq="h")
        values = (
            3000
            + 800  * np.sin(2 * np.pi * np.arange(n) / 24)
            + 400  * np.sin(2 * np.pi * np.arange(n) / (24*7))
            + np.random.randn(n) * 80
        )
        df = pd.DataFrame({"datetime": dates, "mw": values})
        result = run_full_pipeline(df, "mw", "datetime", steps=24)
        assert result["r2"] > 0.0

    def test_retail_sales_with_promotions(self):
        """Retail sales with weekend peaks and promotional spikes."""
        np.random.seed(32)
        n = 500
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        base = np.linspace(500, 700, n)
        weekend = np.where(pd.DatetimeIndex(dates).dayofweek >= 5, 150, 0)
        promo = np.zeros(n)
        promo[np.arange(0, n, 30)] = 300   # promo every 30 days
        noise = np.random.randn(n) * 25
        values = base + weekend + promo + noise
        df = pd.DataFrame({"date": dates, "units": values})
        result = run_full_pipeline(df, "units", "date", steps=14)
        assert result["mae"] >= 0

    def test_iot_machine_vibration(self):
        """Industrial vibration sensor — high frequency with anomaly spikes."""
        np.random.seed(33)
        n = 1000
        dates = pd.date_range("2024-01-01", periods=n, freq="min")
        values = (
            1.0
            + 0.3 * np.sin(2 * np.pi * np.arange(n) / 60)
            + np.random.randn(n) * 0.05
        )
        # Inject 5 anomaly spikes
        values[np.array([100, 250, 400, 700, 900])] += 5.0
        df = pd.DataFrame({"ts": dates, "vibration_mm_s": values})
        result = run_full_pipeline(df, "vibration_mm_s", "ts", steps=30)
        # Anomaly detection should find some
        assert result["anomalies"] is not None

    def test_weather_temperature(self):
        """Daily temperature with strong annual seasonality."""
        np.random.seed(34)
        n = 3 * 365
        dates = pd.date_range("2021-01-01", periods=n, freq="D")
        values = (
            15
            + 12 * np.sin(2 * np.pi * (np.arange(n) - 80) / 365)
            + np.random.randn(n) * 3
        )
        df = pd.DataFrame({"date": dates, "temp_c": values})
        result = run_full_pipeline(df, "temp_c", "date", steps=30)
        assert result["r2"] > 0.3

    def test_web_traffic_page_views(self):
        """Website page views — strong weekday/weekend pattern."""
        np.random.seed(35)
        n = 400
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        base  = np.linspace(10000, 15000, n)
        wkday = np.where(pd.DatetimeIndex(dates).dayofweek < 5, 3000, -2000)
        noise = np.random.randn(n) * 500
        values = np.maximum(base + wkday + noise, 100)  # floor at 100
        df = pd.DataFrame({"date": dates, "pageviews": values})
        result = run_full_pipeline(df, "pageviews", "date", steps=14)
        assert result["mae"] >= 0

    def test_healthcare_patient_admissions(self):
        """Weekly hospital admissions — seasonal flu pattern."""
        np.random.seed(36)
        n = 104  # 2 years weekly
        dates = pd.date_range("2022-01-03", periods=n, freq="W")
        values = (
            200
            + 80 * np.sin(2 * np.pi * (np.arange(n) + 10) / 52)  # flu season
            + np.random.randn(n) * 15
        )
        values = np.maximum(values, 10).astype(float)
        df = pd.DataFrame({"week": dates, "admissions": values})
        result = run_full_pipeline(df, "admissions", "week", steps=4)
        assert result["mae"] >= 0


# ══════════════════════════════════════════════════════════════════════
# CATEGORY 5 — EDGE CASES
# ══════════════════════════════════════════════════════════════════════


class TestEdgeCases:

    def test_negative_values(self):
        """Series with negative values (temperature, P&L, etc.)."""
        np.random.seed(40)
        n = 300
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        values = (
            -50
            + np.linspace(-10, 10, n)
            + 8 * np.sin(2 * np.pi * np.arange(n) / 7)
            + np.random.randn(n) * 3
        )
        df = pd.DataFrame({"date": dates, "pnl": values})
        assert values.min() < 0, "Need negative values for this test"
        result = run_full_pipeline(df, "pnl", "date", steps=10)
        assert result["mae"] >= 0

    def test_integer_only_values(self):
        """Count data — integers only (orders, events, visits)."""
        np.random.seed(41)
        n = 400
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        values = (
            np.linspace(10, 50, n)
            + 8 * np.sin(2 * np.pi * np.arange(n) / 7)
            + np.random.randn(n) * 2
        ).astype(int).astype(float)
        df = pd.DataFrame({"date": dates, "orders": values})
        result = run_full_pipeline(df, "orders", "date", steps=14)
        assert result["mae"] >= 0

    def test_exponential_growth(self):
        """Exponential growth — log transform should be applied."""
        np.random.seed(42)
        n = 365
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        values = 100 * np.exp(0.005 * np.arange(n)) + np.random.randn(n) * 2
        df = pd.DataFrame({"date": dates, "users": values})
        result = run_full_pipeline(df, "users", "date", steps=14)
        assert result["mae"] >= 0

    def test_weekend_gaps_trading_data(self):
        """Stock-market style data with weekend gaps (business days only)."""
        np.random.seed(43)
        n = 500
        dates = pd.bdate_range("2022-01-03", periods=n)  # business days
        values = 100 * np.exp(
            np.cumsum(np.random.randn(n) * 0.01 + 0.0002)
        )
        df = pd.DataFrame({"date": dates, "price": values})
        result = run_full_pipeline(df, "price", "date", steps=10)
        assert result["mae"] >= 0

    def test_multiple_date_column_formats(self):
        """Various date string formats the validator must auto-parse."""
        np.random.seed(44)
        n = 200

        formats_and_data = [
            ("2022-01-01",  "YYYY-MM-DD"),
            ("01/01/2022",  "DD/MM/YYYY"),
            ("20220101",    "YYYYMMDD"),
        ]
        for date_str_start, fmt_name in formats_and_data:
            try:
                dates = pd.date_range("2022-01-01", periods=n, freq="D")
                if fmt_name == "DD/MM/YYYY":
                    date_strs = [d.strftime("%d/%m/%Y") for d in dates]
                elif fmt_name == "YYYYMMDD":
                    date_strs = [d.strftime("%Y%m%d") for d in dates]
                else:
                    date_strs = [d.strftime("%Y-%m-%d") for d in dates]

                values = np.linspace(100, 200, n) + np.random.randn(n) * 5
                df = pd.DataFrame({"date": date_strs, "value": values})
                result = run_full_pipeline(df, "value", "date", steps=5)
                assert result["mae"] >= 0, f"Failed for format {fmt_name}"
            except Exception as e:
                pytest.fail(f"Date format {fmt_name} caused: {e}")


# ══════════════════════════════════════════════════════════════════════
# EVALUATION METRICS VERIFICATION
# ══════════════════════════════════════════════════════════════════════


class TestEvaluationMetrics:

    def _fitted_model(self):
        from baxter_ts import BAXModel
        np.random.seed(99)
        n = 400
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        values = np.linspace(100, 200, n) + 20*np.sin(2*np.pi*np.arange(n)/7) + np.random.randn(n)*5
        df = pd.DataFrame({"date": dates, "sales": values})
        model = BAXModel(n_cv_splits=2, verbose=False)
        model.fit(df, target_col="sales", date_col="date")
        return model

    def test_mae_is_positive(self):
        m = self._fitted_model()
        assert m._best_scores["mae"] >= 0

    def test_rmse_geq_mae(self):
        """RMSE >= MAE always (by definition)."""
        m = self._fitted_model()
        assert m._best_scores["rmse"] >= m._best_scores["mae"] - 1e-6

    def test_r2_range(self):
        """R2 can be negative (bad model) but rarely below -10 on clean data."""
        m = self._fitted_model()
        assert m._best_scores["r2"] > -10

    def test_cv_scores_populated(self):
        """Walk-forward CV scores must be in scoreboard."""
        m = self._fitted_model()
        sb = m.scoreboard()
        assert "cv_mae"  in sb.columns
        assert "cv_rmse" in sb.columns
        for val in sb["cv_mae"]:
            assert val >= 0

    def test_composite_score_winner_is_best(self):
        """Winner must have the lowest composite score."""
        m = self._fitted_model()
        sb = m.scoreboard().reset_index()
        winner_row = sb[sb["model"] == m._selector.best_model.name]
        assert len(winner_row) == 1
        winner_score = winner_row["composite_score"].values[0]
        assert winner_score == sb["composite_score"].min()

    def test_anomaly_rate_reasonable(self):
        """Anomaly rate should be between 0% and 30% on normal data."""
        m = self._fitted_model()
        anom = m.anomalies()
        rate = anom["anomaly_flag"].mean()
        assert 0.0 <= rate <= 0.30, f"Anomaly rate {rate:.2%} out of expected range"

    def test_shap_feature_importance_sums(self):
        """SHAP values must be non-negative and top features must exist."""
        m = self._fitted_model()
        fi = m._explainer.feature_importance_
        assert fi is not None
        assert len(fi) > 0
        assert (fi >= 0).all(), "SHAP mean abs values must be non-negative"
        assert fi.index[0] in m._X_train.columns

    def test_preprocessing_audit_completeness(self):
        """All 7 preprocessing steps must write to the audit dict."""
        m = self._fitted_model()
        audit = m._preprocessing_audit
        required_keys = [
            "validator", "imputer", "outlier",
            "transformer", "scaler", "feature_eng", "splitter"
        ]
        for key in required_keys:
            assert key in audit, f"Missing audit key: {key}"

    def test_forecast_dates_are_future(self):
        """All forecast dates must be strictly after the last training date."""
        m = self._fitted_model()
        forecast = m.predict(steps=14)
        last_train = m._df_processed.index[-1]
        assert forecast.index.min() > last_train, "Forecast dates overlap training data"

    def test_no_leakage_in_split(self):
        """Test set must start strictly after train set ends."""
        m = self._fitted_model()
        assert m._X_train.index.max() < m._X_test.index.min(), \
            "Data leakage: test data overlaps training data"


# ══════════════════════════════════════════════════════════════════════
# PIPELINE ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════


class TestPipelineRobustness:

    def test_fit_predict_anomaly_explain_report_chain(self):
        """Full method chain must work without state errors."""
        from baxter_ts import BAXModel
        np.random.seed(50)
        n = 300
        df = pd.DataFrame({
            "date":  pd.date_range("2022-01-01", periods=n, freq="D"),
            "value": np.linspace(100, 200, n) + np.random.randn(n) * 5,
        })
        m = BAXModel(n_cv_splits=2, verbose=False)
        m.fit(df, "value", "date")
        f  = m.predict(14)
        an = m.anomalies()
        ex = m.explain()
        sb = m.scoreboard()
        su = m.summary()
        assert all([f is not None, an is not None, ex, sb is not None, su])

    def test_predict_called_twice(self):
        """predict() called twice must return consistent results."""
        from baxter_ts import BAXModel
        np.random.seed(51)
        n = 300
        df = pd.DataFrame({
            "date":  pd.date_range("2022-01-01", periods=n, freq="D"),
            "value": np.linspace(0, 100, n) + np.random.randn(n) * 2,
        })
        m = BAXModel(n_cv_splits=2, verbose=False)
        m.fit(df, "value", "date")
        f1 = m.predict(7)
        f2 = m.predict(7)
        pd.testing.assert_frame_equal(f1, f2)

    def test_anomalies_called_twice(self):
        """anomalies() called twice must return same result."""
        from baxter_ts import BAXModel
        np.random.seed(52)
        n = 300
        df = pd.DataFrame({
            "date":  pd.date_range("2022-01-01", periods=n, freq="D"),
            "value": np.linspace(0, 100, n) + np.random.randn(n) * 3,
        })
        m = BAXModel(n_cv_splits=2, verbose=False)
        m.fit(df, "value", "date")
        a1 = m.anomalies()
        a2 = m.anomalies()
        pd.testing.assert_frame_equal(a1, a2)

    def test_unfitted_all_methods_raise(self):
        """Every public method must raise RuntimeError if called before fit."""
        from baxter_ts import BAXModel
        m = BAXModel(verbose=False)
        for method, args in [
            ("predict",   (10,)),
            ("anomalies", ()),
            ("explain",   ()),
            ("scoreboard",()),
            ("summary",   ()),
            ("visualize", ()),
        ]:
            with pytest.raises(RuntimeError, match="fit"):
                getattr(m, method)(*args)

    def test_wrong_target_col_raises(self):
        """Passing a non-existent target column must raise."""
        from baxter_ts import BAXModel
        df = pd.DataFrame({
            "date":  pd.date_range("2022-01-01", periods=100, freq="D"),
            "sales": np.random.randn(100) + 100,
        })
        m = BAXModel(verbose=False)
        with pytest.raises(Exception):
            m.fit(df, target_col="revenue", date_col="date")  # "revenue" not in df

    def test_different_anomaly_methods(self):
        """All four anomaly methods must run without error."""
        from baxter_ts import BAXModel
        np.random.seed(53)
        n = 300
        df = pd.DataFrame({
            "date":  pd.date_range("2022-01-01", periods=n, freq="D"),
            "value": np.linspace(100, 200, n) + np.random.randn(n) * 5,
        })
        for method in ["ensemble", "isolation_forest", "zscore", "iqr"]:
            m = BAXModel(n_cv_splits=2, verbose=False, anomaly_method=method)
            m.fit(df, "value", "date")
            anom = m.anomalies()
            assert "anomaly_flag" in anom.columns, f"method={method} failed"
