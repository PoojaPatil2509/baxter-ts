"""
Step 8: Automated time-series feature engineering.
Covers lags, rolling stats, EWM, calendar, Fourier terms, holiday flags.
Works for any frequency from sub-minute to yearly.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class TimeSeriesFeatureEngineer:
    PERIOD_MAP = {"min": 1440, "h": 24, "D": 365.25, "W": 52, "MS": 12, "YS": 1, "Q": 4}

    def __init__(
        self,
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        add_calendar: bool = True,
        add_fourier: bool = True,
        fourier_order: int = 3,
        add_holidays: bool = True,
    ):
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.add_calendar = add_calendar
        self.add_fourier = add_fourier
        self.fourier_order = fourier_order
        self.add_holidays = add_holidays
        self.feature_names_: List[str] = []
        self.audit: dict = {}

    def fit_transform(self, df: pd.DataFrame, target_col: str, freq: str = "D") -> pd.DataFrame:
        df = df.copy()
        original_cols = set(df.columns)

        self.lags = self.lags or self._default_lags(freq)
        self.rolling_windows = self.rolling_windows or self._default_windows(freq)

        # Lag features
        for lag in self.lags:
            df[f"lag_{lag}"] = df[target_col].shift(lag)

        # Rolling statistics (shift(1) to avoid data leakage)
        for win in self.rolling_windows:
            base = df[target_col].shift(1)
            df[f"roll_mean_{win}"] = base.rolling(win, min_periods=1).mean()
            df[f"roll_std_{win}"]  = base.rolling(win, min_periods=2).std()
            df[f"roll_min_{win}"]  = base.rolling(win, min_periods=1).min()
            df[f"roll_max_{win}"]  = base.rolling(win, min_periods=1).max()
            df[f"roll_range_{win}"] = df[f"roll_max_{win}"] - df[f"roll_min_{win}"]

        # Exponentially weighted mean
        shifted = df[target_col].shift(1)
        df["ewm_02"] = shifted.ewm(alpha=0.2, adjust=False).mean()
        df["ewm_05"] = shifted.ewm(alpha=0.5, adjust=False).mean()
        df["ewm_08"] = shifted.ewm(alpha=0.8, adjust=False).mean()

        # Momentum
        df["pct_change_1"] = df[target_col].pct_change(1).replace([np.inf, -np.inf], np.nan)
        if len(df) > 7:
            df["pct_change_7"] = df[target_col].pct_change(7).replace([np.inf, -np.inf], np.nan)

        # Numeric time index (captures global trend)
        df["time_idx"] = np.arange(len(df))

        if self.add_calendar:
            df = self._add_calendar_features(df, freq)

        if self.add_fourier:
            df = self._add_fourier_features(df, freq)

        if self.add_holidays:
            df = self._add_holiday_flags(df)

        new_cols = list(set(df.columns) - original_cols)
        self.feature_names_ = new_cols
        self.audit.update({
            "lags": self.lags,
            "rolling_windows": self.rolling_windows,
            "total_features_added": len(new_cols),
            "feature_names": new_cols,
        })
        return df

    def _default_lags(self, freq: str) -> List[int]:
        return {
            "T":  [1, 5, 15, 30, 60],
            "H":  [1, 3, 6, 12, 24, 48],
            "D":  [1, 7, 14, 21, 30],
            "W":  [1, 4, 8, 13, 26, 52],
            "MS": [1, 3, 6, 9, 12],
            "Q":  [1, 2, 4],
            "YS": [1, 2, 3],
        }.get(freq, [1, 7, 14, 30])

    def _default_windows(self, freq: str) -> List[int]:
        return {
            "T":  [5, 15, 60],
            "H":  [6, 12, 24],
            "D":  [7, 14, 30],
            "W":  [4, 13, 26],
            "MS": [3, 6, 12],
            "Q":  [2, 4],
            "YS": [2, 3],
        }.get(freq, [7, 14, 30])

    def _add_calendar_features(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        idx = df.index
        df["dayofweek"]    = idx.dayofweek
        df["month"]        = idx.month
        df["quarter"]      = idx.quarter
        df["year"]         = idx.year
        df["dayofyear"]    = idx.dayofyear
        df["is_weekend"]   = (idx.dayofweek >= 5).astype(int)
        try:
            df["week_of_year"] = idx.isocalendar().week.astype(int).values
        except Exception:
            df["week_of_year"] = idx.week
        if freq in ["h", "min"]:
            df["hour"] = idx.hour
            df["minute"] = idx.minute
            df["is_business_hour"] = ((idx.hour >= 9) & (idx.hour <= 17)).astype(int)
        return df

    def _add_fourier_features(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        period = self.PERIOD_MAP.get(freq, 365.25)
        t = np.arange(len(df))
        for k in range(1, self.fourier_order + 1):
            df[f"sin_k{k}"] = np.sin(2 * np.pi * k * t / period)
            df[f"cos_k{k}"] = np.cos(2 * np.pi * k * t / period)
        return df

    def _add_holiday_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df["is_month_start"]  = df.index.is_month_start.astype(int)
            df["is_month_end"]    = df.index.is_month_end.astype(int)
            df["is_quarter_end"]  = df.index.is_quarter_end.astype(int)
            df["is_year_end"]     = df.index.is_year_end.astype(int)
        except Exception:
            pass
        return df
