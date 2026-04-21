"""Abstract base class for all baxter-ts models."""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, Optional


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.abs(y_true) > 1e-8
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


class BaseTimeSeriesModel(ABC):
    def __init__(self, name: str):
        self.name = name
        self._model = None
        self.feature_cols_: list = []
        self.cv_scores_: Dict[str, float] = {}
        self.test_scores_: Dict[str, float] = {}
        self.is_fitted: bool = False

    @abstractmethod
    def _build_model(self):
        """Return an unfitted sklearn-compatible estimator."""
        ...

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            cv_splits: Optional[list] = None) -> "BaseTimeSeriesModel":
        self._model = self._build_model()
        self.feature_cols_ = X_train.columns.tolist()

        # Walk-forward cross-validation
        if cv_splits:
            cv_maes, cv_rmses = [], []
            X_arr = X_train.values
            y_arr = y_train.values
            for tr_idx, val_idx in cv_splits:
                self._model.fit(X_arr[tr_idx], y_arr[tr_idx])
                preds = self._model.predict(X_arr[val_idx])
                cv_maes.append(mean_absolute_error(y_arr[val_idx], preds))
                cv_rmses.append(np.sqrt(mean_squared_error(y_arr[val_idx], preds)))
            self.cv_scores_ = {
                "cv_mae": round(float(np.mean(cv_maes)), 4),
                "cv_rmse": round(float(np.mean(cv_rmses)), 4),
            }

        # Final fit on full training data
        self._model.fit(X_train.values, y_train.values)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} is not fitted yet.")
        return self._model.predict(X[self.feature_cols_].values)

    def score(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        preds = self.predict(X_test)
        y_true = y_test.values
        mae  = mean_absolute_error(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        mape = _safe_mape(y_true, preds)
        r2   = float(1 - np.sum((y_true - preds) ** 2) / (np.sum((y_true - np.mean(y_true)) ** 2) + 1e-9))
        self.test_scores_ = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 4),
            "r2": round(r2, 4),
        }
        return self.test_scores_

    @property
    def composite_score(self) -> float:
        """Lower is better — weighted rank across MAE + RMSE + MAPE."""
        s = self.test_scores_
        if not s:
            return float("inf")
        mae  = s.get("mae", float("inf"))
        rmse = s.get("rmse", float("inf"))
        mape = s.get("mape", float("inf"))
        mape = mape if not np.isnan(mape) else 100.0
        return round(0.4 * mae + 0.4 * rmse + 0.2 * (mape / 100), 7)
