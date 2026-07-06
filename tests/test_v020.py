"""
v0.2.0 feature tests.

Covers the release headliners:
  - predict() returns forecasts in ORIGINAL units (not scaled/diffed space)
  - prediction intervals (forecast, lower, upper)
  - SeasonalNaive baseline competes on the scoreboard
  - save()/load() persistence roundtrip
  - stability: hourly freq lags, constant target, short series, clear errors
"""

import numpy as np
import pandas as pd
import pytest

from baxter_ts import BAXModel


def make_series(n=200, start=100.0, slope=1.0, noise=2.0, freq="D", seed=0):
    """Trending + weekly-seasonal series in a realistic value range."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq=freq)
    y = (
        start
        + slope * np.arange(n)
        + 10 * np.sin(2 * np.pi * np.arange(n) / 7)
        + rng.normal(0, noise, n)
    )
    return pd.DataFrame({"date": dates, "y": y})


@pytest.fixture(scope="module")
def fitted():
    model = BAXModel(verbose=False)
    model.fit(make_series(), target_col="y", date_col="date")
    return model


# ----------------------------------------------------------------------
# Original-unit forecasts + intervals
# ----------------------------------------------------------------------

def test_forecast_in_original_units(fitted):
    """
    The series runs ~100 → ~300. v0.1.x returned scaled/differenced values
    (magnitude ~0.01-1); v0.2.0 must return values in the data's own range.
    """
    fc = fitted.predict(steps=14)
    assert fc["forecast"].abs().mean() > 50, (
        "forecast looks like scaled space, not original units"
    )
    assert fc["forecast"].between(100, 800).all()


def test_predict_returns_intervals(fitted):
    fc = fitted.predict(steps=10)
    assert list(fc.columns) == ["forecast", "lower", "upper"]
    assert (fc["lower"] <= fc["forecast"]).all()
    assert (fc["forecast"] <= fc["upper"]).all()
    # Band must have nonzero width
    assert (fc["upper"] - fc["lower"]).mean() > 0


def test_original_scale_metrics_sane(fitted):
    """Exact original metrics: a clean trend must not show ~99% MAPE."""
    s = fitted.summary()
    assert s["test_mape"] is not None
    assert s["test_mape"] < 25
    assert s["test_mae"] < 60  # series scale is ~100-300


def test_display_series_original_units(fitted):
    y_disp, pred_disp = fitted._display_test_series()
    assert y_disp.abs().mean() > 50
    assert np.abs(pred_disp).mean() > 50


# ----------------------------------------------------------------------
# Baseline
# ----------------------------------------------------------------------

def test_baseline_in_scoreboard(fitted):
    board = fitted.scoreboard()
    assert "SeasonalNaive" in board.index


def test_baseline_can_be_disabled():
    model = BAXModel(include_baseline=False, verbose=False)
    model.fit(make_series(n=120), target_col="y", date_col="date")
    assert "SeasonalNaive" not in model.scoreboard().index


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------

def test_save_load_roundtrip(fitted, tmp_path):
    path = fitted.save(str(tmp_path / "model"))
    loaded = BAXModel.load(path)

    fc_a = fitted.predict(steps=7)
    fc_b = loaded.predict(steps=7)
    pd.testing.assert_frame_equal(fc_a, fc_b)

    # explain() must survive the roundtrip (narrative is kept)
    assert loaded.explain() == fitted._bax_narrative
    assert loaded.summary()["best_model"] == fitted.summary()["best_model"]


# ----------------------------------------------------------------------
# Stability across dataset types
# ----------------------------------------------------------------------

def test_hourly_frequency_gets_hourly_lags():
    """v0.1.x freq-alias bug: hourly data received daily lag defaults."""
    model = BAXModel(verbose=False)
    model.fit(make_series(n=300, freq="h"), target_col="y", date_col="date")
    assert 24 in model._feat_eng.lags


def test_constant_series_does_not_crash():
    dates = pd.date_range("2023-01-01", periods=80, freq="D")
    df = pd.DataFrame({"date": dates, "y": np.full(80, 42.0)})
    model = BAXModel(verbose=False)
    model.fit(df, target_col="y", date_col="date")
    fc = model.predict(steps=5)
    assert np.allclose(fc["forecast"], 42.0, atol=5.0)


def test_short_series_fits():
    model = BAXModel(verbose=False)
    model.fit(make_series(n=30), target_col="y", date_col="date")
    fc = model.predict(steps=3)
    assert len(fc) == 3


def test_too_few_rows_raises_clear_error():
    with pytest.raises(ValueError, match="rows"):
        BAXModel(verbose=False).fit(
            make_series(n=10), target_col="y", date_col="date"
        )


def test_weekly_anchored_freq():
    """'W-SUN' style anchored aliases must normalize cleanly."""
    model = BAXModel(verbose=False)
    model.fit(make_series(n=160, freq="W-SUN"), target_col="y", date_col="date")
    assert model._freq == "W"
    fc = model.predict(steps=4)
    assert len(fc) == 4


# ----------------------------------------------------------------------
# Tuning (smoke test — small data to keep runtime down)
# ----------------------------------------------------------------------

def test_tune_smoke():
    model = BAXModel(tune=True, verbose=False)
    model.fit(make_series(n=100), target_col="y", date_col="date")
    assert model.summary()["best_model"] is not None
    assert model._selector.audit["tuned"] is True


# ----------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------

def test_normalize_freq():
    from baxter_ts.utils import normalize_freq
    assert normalize_freq("T") == "min"
    assert normalize_freq("min") == "min"
    assert normalize_freq("15min") == "min"
    assert normalize_freq("H") == "h"
    assert normalize_freq("h") == "h"
    assert normalize_freq("B") == "D"
    assert normalize_freq("W-SUN") == "W"
    assert normalize_freq("ME") == "MS"
    assert normalize_freq("M") == "MS"
    assert normalize_freq("Q-DEC") == "Q"
    assert normalize_freq("A") == "YS"
    assert normalize_freq(None) == "D"
    assert normalize_freq("garbage") == "D"
