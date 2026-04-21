"""Step 7: Scaling and normalisation — auto-selects scaler by distribution shape."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


class TimeSeriesScaler:
    SCALERS = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "robust": RobustScaler,
    }

    def __init__(self, strategy: str = "auto"):
        self.strategy = strategy
        self._scaler = None
        self._chosen: str = ""
        self._feature_cols: list = []
        self.audit: dict = {}

    def fit_transform(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        df = df.copy()
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        self._feature_cols = num_cols

        chosen = self._auto_select(df[target_col]) if self.strategy == "auto" else self.strategy
        self._chosen = chosen

        scaler_cls = self.SCALERS.get(chosen, RobustScaler)
        self._scaler = scaler_cls()
        df[num_cols] = self._scaler.fit_transform(df[num_cols])

        self.audit["scaler_used"] = chosen
        self.audit["columns_scaled"] = len(num_cols)
        return df

    def inverse_transform_target(self, values: np.ndarray, target_col: str, df_ref: pd.DataFrame) -> np.ndarray:
        """Reverse-scale predictions back to original value range."""
        if self._scaler is None:
            return values
        try:
            col_idx = self._feature_cols.index(target_col)
            dummy = np.zeros((len(values), len(self._feature_cols)))
            dummy[:, col_idx] = values.ravel()
            inversed = self._scaler.inverse_transform(dummy)
            return inversed[:, col_idx]
        except Exception:
            return values

    def _auto_select(self, series: pd.Series) -> str:
        try:
            skew = abs(float(series.skew()))
            pct_outliers = ((series - series.mean()).abs() > 3 * series.std()).mean()
            if pct_outliers > 0.05 or skew > 2.0:
                return "robust"
            elif skew < 0.5:
                return "standard"
            else:
                return "minmax"
        except Exception:
            return "robust"
