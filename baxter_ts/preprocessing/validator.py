"""
Step 1: Datetime parsing and index validation
Step 2: Frequency inference and gap detection
"""

import pandas as pd
import numpy as np
from typing import Optional, List


class DatetimeValidator:
    COMMON_DATE_FORMATS = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
        "%Y%m%d", "%d-%m-%Y", "%Y/%m/%d",
    ]

    def __init__(self):
        self.detected_freq: Optional[str] = None
        self.date_col: Optional[str] = None
        self.audit: dict = {}

    def fit_transform(
        self,
        df: pd.DataFrame,
        date_col: Optional[str] = None,
        target_col: Optional[str] = None,
    ) -> pd.DataFrame:
        df = df.copy()

        if date_col:
            self.date_col = date_col
        else:
            self.date_col = self._auto_detect_date_col(df)

        if self.date_col and self.date_col in df.columns:
            df[self.date_col] = self._parse_datetime(df[self.date_col])
            df = df.set_index(self.date_col)
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "No datetime column found. Pass date_col='your_col' explicitly."
            )

        df = df.sort_index()

        dupes = df.index.duplicated().sum()
        if dupes > 0:
            df = df[~df.index.duplicated(keep="last")]
            self.audit["duplicate_timestamps_removed"] = int(dupes)

        self.detected_freq = self._infer_frequency(df)
        gaps = self._find_gaps(df, self.detected_freq)

        self.audit.update(
            {
                "date_col": self.date_col,
                "inferred_frequency": self.detected_freq,
                "total_rows": len(df),
                "date_range": f"{df.index.min()} to {df.index.max()}",
                "gap_count": len(gaps),
                "gaps_sample": gaps[:5],
            }
        )
        return df

    def _auto_detect_date_col(self, df: pd.DataFrame) -> Optional[str]:
        priority = ["date", "datetime", "timestamp", "time", "ds", "Date", "Datetime", "TIME", "index"]
        for name in priority:
            if name in df.columns:
                return name
        for col in df.columns:
            try:
                pd.to_datetime(df[col].head(10), infer_datetime_format=True)
                return col
            except Exception:
                continue
        return None

    def _parse_datetime(self, series: pd.Series) -> pd.Series:
        try:
            return pd.to_datetime(series, infer_datetime_format=True)
        except Exception:
            pass
        for fmt in self.COMMON_DATE_FORMATS:
            try:
                return pd.to_datetime(series, format=fmt)
            except Exception:
                continue
        raise ValueError(f"Could not parse datetime from column '{series.name}'.")

    def _infer_frequency(self, df: pd.DataFrame) -> str:
        try:
            freq = pd.infer_freq(df.index)
            if freq:
                return freq
        except Exception:
            pass
        gaps = df.index.to_series().diff().dropna()
        median_gap = gaps.median()
        if median_gap <= pd.Timedelta(minutes=1):
            return "min"
        elif median_gap <= pd.Timedelta(hours=1):
            return "h"
        elif median_gap <= pd.Timedelta(days=1):
            return "D"
        elif median_gap <= pd.Timedelta(days=7):
            return "W"
        elif median_gap <= pd.Timedelta(days=31):
            return "MS"
        elif median_gap <= pd.Timedelta(days=366):
            return "YS"
        return "D"

    def _find_gaps(self, df: pd.DataFrame, freq: str) -> List[str]:
        try:
            full_range = pd.date_range(df.index.min(), df.index.max(), freq=freq)
            missing = full_range.difference(df.index)
            return [str(ts) for ts in missing[:20]]
        except Exception:
            return []
