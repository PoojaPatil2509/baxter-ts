"""Step 9: Temporal train/test split with walk-forward cross-validation."""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, List


class TemporalSplitter:
    def __init__(self, test_size: float = 0.2, n_splits: int = 5):
        self.test_size = test_size
        self.n_splits = n_splits
        self.audit: dict = {}

    def split(
        self, df: pd.DataFrame, target_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Simple last-N-rows test split — no shuffle, respects time."""
        n = len(df)
        test_n = max(1, int(n * self.test_size))
        train_df = df.iloc[: n - test_n]
        test_df = df.iloc[n - test_n :]

        feature_cols = [c for c in df.columns if c != target_col]
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        self.audit.update(
            {
                "train_size": len(train_df),
                "test_size": len(test_df),
                "train_pct": round(len(train_df) / n * 100, 1),
                "test_pct": round(len(test_df) / n * 100, 1),
                "train_start": str(train_df.index.min()),
                "train_end": str(train_df.index.max()),
                "test_start": str(test_df.index.min()),
                "test_end": str(test_df.index.max()),
                "n_features": len(feature_cols),
            }
        )
        return X_train, X_test, y_train, y_test

    def cv_splits(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple]:
        """Walk-forward CV folds — no future data leaks into past folds."""
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        return list(tscv.split(X))
