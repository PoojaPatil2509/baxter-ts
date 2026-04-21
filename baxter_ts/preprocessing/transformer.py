"""
Step 5: Stationarity testing and transformation (ADF/KPSS, differencing, log, Box-Cox)
Step 6: STL decomposition — extracts trend, seasonal, residual as model features
"""

import warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL


class StationarityTransformer:
    PERIOD_MAP = {"min": 60, "h": 24, "D": 7, "W": 52, "MS": 12, "YS": 1, "Q": 4}

    def __init__(self, strategy: str = "auto", max_diffs: int = 2):
        self.strategy = strategy
        self.max_diffs = max_diffs
        self.n_diffs_applied: int = 0
        self.log_applied: bool = False
        self.decomposition = None
        self.audit: dict = {}

    def fit_transform(
        self, df: pd.DataFrame, target_col: str, freq: str = "D"
    ) -> pd.DataFrame:
        df = df.copy()
        series = df[target_col].dropna()

        is_stationary, adf_p, kpss_p = self._test_stationarity(series)
        self.audit["adf_pvalue"] = round(float(adf_p), 4)
        self.audit["kpss_pvalue"] = round(float(kpss_p), 4)
        self.audit["was_stationary"] = bool(is_stationary)

        if not is_stationary:
            if (df[target_col] > 0).all() and float(series.skew()) > 1.0:
                df[target_col] = np.log1p(df[target_col])
                self.log_applied = True
                self.audit["log_transform_applied"] = True

            for d in range(1, self.max_diffs + 1):
                test_series = df[target_col].diff(d).dropna()
                if len(test_series) < 10:
                    break
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stat, p, *_ = adfuller(test_series, autolag="AIC")
                if p < 0.05:
                    df[target_col] = df[target_col].diff(d)
                    df.dropna(subset=[target_col], inplace=True)
                    self.n_diffs_applied = d
                    break

        self.audit["diffs_applied"] = self.n_diffs_applied
        df = self._decompose(df, target_col, freq)
        return df

    def _test_stationarity(self, series: pd.Series):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                adf_stat, adf_p, *_ = adfuller(series, autolag="AIC")
            except Exception:
                adf_p = 1.0
            try:
                _, kpss_p, *_ = kpss(series, regression="c", nlags="auto")
            except Exception:
                kpss_p = 0.05
        return (adf_p < 0.05) and (kpss_p > 0.05), adf_p, kpss_p

    def _decompose(self, df: pd.DataFrame, target_col: str, freq: str) -> pd.DataFrame:
        try:
            period = self.PERIOD_MAP.get(freq, 7)
            min_len = period * 3
            clean = df[target_col].dropna()
            if len(clean) < min_len:
                self.audit["stl_decomposition"] = False
                self.audit["stl_skip_reason"] = f"need {min_len} rows, got {len(clean)}"
                return df
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stl = STL(clean, period=period, robust=True)
                result = stl.fit()
            df["trend_component"] = result.trend
            df["seasonal_component"] = result.seasonal
            df["residual_component"] = result.resid
            self.decomposition = result
            self.audit["stl_decomposition"] = True
        except Exception as e:
            self.audit["stl_decomposition"] = False
            self.audit["stl_error"] = str(e)
        return df
