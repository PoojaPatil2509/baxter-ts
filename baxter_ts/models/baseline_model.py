"""
Seasonal-naive baseline: forecast = the value one season ago.

It joins the AutoML scoreboard so the tree models have to prove they beat
the simplest credible forecast. If SeasonalNaive wins, the series is close
to a seasonal random walk and users should treat ML gains with skepticism —
the BAX narrative calls this out.

Implementation detail: the pipeline's feature matrix already contains lag
columns of the (transformed) target, so the baseline simply reads the lag
column closest to one seasonal period. This keeps it inside the existing
BaseTimeSeriesModel interface — same predict(X), same scoring, same
composite score — with zero special-casing in the selector loop.
"""

import re
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Optional

from baxter_ts.models.base_model import BaseTimeSeriesModel

# Seasonal period per canonical frequency (see baxter_ts.utils.normalize_freq)
SEASON_BY_FREQ = {"min": 60, "h": 24, "D": 7, "W": 52, "MS": 12, "Q": 4, "YS": 1}


class SeasonalNaiveModel(BaseTimeSeriesModel):
    def __init__(self, freq: str = "D", random_state: int = 42):
        super().__init__("SeasonalNaive")
        self.freq = freq
        self.random_state = random_state
        self.lag_col_: Optional[str] = None

    def _build_model(self):
        return None  # no underlying estimator

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            cv_splits: Optional[list] = None) -> "SeasonalNaiveModel":
        self.feature_cols_ = X_train.columns.tolist()
        self.lag_col_ = self._pick_lag_col(X_train.columns)
        if self.lag_col_ is None:
            raise RuntimeError("SeasonalNaive needs at least one lag_* feature.")

        if cv_splits:
            col = X_train[self.lag_col_].values
            y_arr = y_train.values
            maes, rmses = [], []
            for _, val_idx in cv_splits:
                preds = col[val_idx]
                maes.append(mean_absolute_error(y_arr[val_idx], preds))
                rmses.append(np.sqrt(mean_squared_error(y_arr[val_idx], preds)))
            self.cv_scores_ = {
                "cv_mae": round(float(np.mean(maes)), 4),
                "cv_rmse": round(float(np.mean(rmses)), 4),
            }

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} is not fitted yet.")
        return X[self.lag_col_].values.astype(float)

    def _pick_lag_col(self, columns) -> Optional[str]:
        """Lag column closest to one seasonal period; ties go to the shorter lag."""
        season = SEASON_BY_FREQ.get(self.freq, 7)
        lags = {}
        for c in columns:
            m = re.fullmatch(r"lag_(\d+)", str(c))
            if m:
                lags[int(m.group(1))] = c
        if not lags:
            return None
        best = min(lags, key=lambda n: (abs(n - season), n))
        return lags[best]
