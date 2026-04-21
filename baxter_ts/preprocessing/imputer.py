"""Step 3: Smart missing value imputation respecting temporal order."""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


class TimeSeriesImputer:
    def __init__(self, strategy: str = "auto", knn_neighbors: int = 5):
        self.strategy = strategy
        self.knn_neighbors = knn_neighbors
        self.audit: dict = {}
        self._used_strategy: str = ""

    def fit_transform(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        df = df.copy()
        missing_count = df[target_col].isna().sum()
        missing_pct = missing_count / len(df)

        self.audit["missing_before"] = int(missing_count)
        self.audit["missing_pct"] = round(float(missing_pct * 100), 2)

        if missing_count == 0:
            self.audit["strategy_used"] = "none_needed"
            return df

        chosen = self._auto_select(df, target_col, missing_pct) if self.strategy == "auto" else self.strategy
        self._used_strategy = chosen

        if chosen == "ffill":
            df[target_col] = df[target_col].ffill().bfill()

        elif chosen == "linear_interpolation":
            df[target_col] = df[target_col].interpolate(method="linear", limit_direction="both")

        elif chosen == "time_interpolation":
            df[target_col] = df[target_col].interpolate(method="time", limit_direction="both")

        elif chosen == "seasonal_mean":
            df = self._seasonal_mean_fill(df, target_col)

        elif chosen == "knn":
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            imputer = KNNImputer(n_neighbors=self.knn_neighbors)
            df[num_cols] = imputer.fit_transform(df[num_cols])

        # Safety: fill any leftover NaN
        df[target_col] = df[target_col].ffill().bfill()

        self.audit["strategy_used"] = chosen
        self.audit["missing_after"] = int(df[target_col].isna().sum())
        return df

    def _auto_select(self, df: pd.DataFrame, target_col: str, pct: float) -> str:
        if pct < 0.02:
            return "ffill"
        elif pct < 0.10:
            return "time_interpolation"
        elif pct < 0.25:
            return "seasonal_mean"
        else:
            return "knn"

    def _seasonal_mean_fill(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        idx = df.index
        temp_cols = []
        if hasattr(idx, "hour"):
            df["_hour"] = idx.hour
            df["_dow"] = idx.dayofweek
            means = df.groupby(["_hour", "_dow"])[target_col].transform("mean")
            temp_cols = ["_hour", "_dow"]
        else:
            df["_dow"] = idx.dayofweek
            means = df.groupby("_dow")[target_col].transform("mean")
            temp_cols = ["_dow"]
        df[target_col] = df[target_col].fillna(means)
        df.drop(columns=temp_cols, inplace=True)
        return df
