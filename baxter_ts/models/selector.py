"""
AutoML model selector: trains all candidates, picks the winner by composite score.
Uses walk-forward TimeSeriesSplit — zero data leakage.

Fix v0.1.3:
  Passes y_test_original (raw pre-transformation values from _raw_target_series)
  to model.score() so original-scale MAPE and R² are computed correctly.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from baxter_ts.models.rf_model import RFModel
from baxter_ts.models.xgb_model import XGBModel
from baxter_ts.models.catboost_model import CatModel
from baxter_ts.models.base_model import BaseTimeSeriesModel
from baxter_ts.preprocessing.splitter import TemporalSplitter


class ModelSelector:
    def __init__(self, n_cv_splits: int = 5):
        self.n_cv_splits = n_cv_splits
        self.candidates: List[BaseTimeSeriesModel] = []
        self.scoreboard: List[Dict] = []
        self.best_model: Optional[BaseTimeSeriesModel] = None
        self.audit: dict = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_test_original: Optional[np.ndarray] = None,
    ) -> "ModelSelector":
        self.candidates = [RFModel(), XGBModel(), CatModel()]

        splitter = TemporalSplitter(n_splits=self.n_cv_splits)
        cv_folds = splitter.cv_splits(X_train, y_train)

        self.scoreboard = []
        for model in self.candidates:
            print(f"  Training {model.name}...")
            try:
                model.fit(X_train, y_train, cv_splits=cv_folds)

                # y_pred_scaled — predictions in scaled+differenced space
                y_pred_scaled = model.predict(X_test)

                # Pass y_test_original (raw pre-transformation values).
                # base_model.score() uses _original_scale_metrics() to compute
                # MAE/RMSE/MAPE/R² in the original data space via std-ratio scaling.
                test_scores = model.score(
                    X_test, y_test,
                    y_test_original=y_test_original,
                    y_pred_original=y_pred_scaled,
                )

                # Use original-scale MAPE in scoreboard when available
                orig = model.test_scores_original_
                display_mape = (
                    orig.get("mape", test_scores.get("mape"))
                    if orig else test_scores.get("mape")
                )

                row = {
                    "model":           model.name,
                    **model.cv_scores_,
                    "mae":             test_scores.get("mae"),
                    "rmse":            test_scores.get("rmse"),
                    "mape":            display_mape,
                    "r2":              test_scores.get("r2"),
                    "composite_score": model.composite_score,
                }
                self.scoreboard.append(row)
            except Exception as e:
                print(f"  {model.name} failed: {e}")

        self.scoreboard.sort(key=lambda r: r["composite_score"])

        if not self.scoreboard:
            raise RuntimeError(
                "All models failed. Check that all columns are numeric "
                "and date_col is set correctly."
            )

        best_name = self.scoreboard[0]["model"]
        self.best_model = next(
            m for m in self.candidates if m.name == best_name
        )

        self.audit = {
            "winner":        best_name,
            "scoreboard":    self.scoreboard,
            "n_candidates":  len(self.candidates),
            "failed_models": len(self.candidates) - len(self.scoreboard),
        }

        print(f"\n  Winner: {best_name} "
              f"(composite={self.scoreboard[0]['composite_score']:.4f})")
        return self

    def scoreboard_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.scoreboard).set_index("model")