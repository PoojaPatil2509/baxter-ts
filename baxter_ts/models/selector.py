"""
AutoML model selector: trains all candidates, picks the winner by composite score.
Uses walk-forward TimeSeriesSplit — zero data leakage.

v0.2.0:
  - SeasonalNaive baseline joins the competition so ML models must beat
    the simplest credible forecast to win.
  - Optional hyperparameter tuning (tune=True): small random search per
    model, validated on the tail of the train split. No new dependencies.
  - Original-scale scoring: an invert_fn (TargetInverter) maps scaled
    predictions back to original units so scoreboard metrics are exact.
  - CV folds adapt to short series instead of failing.

Model selection itself still uses the scaled-space composite score — all
candidates share the same transform, so ranks are unchanged and stable.
"""

import random
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional

from baxter_ts.models.rf_model import RFModel
from baxter_ts.models.xgb_model import XGBModel
from baxter_ts.models.catboost_model import CatModel
from baxter_ts.models.baseline_model import SeasonalNaiveModel
from baxter_ts.models.base_model import BaseTimeSeriesModel
from baxter_ts.preprocessing.splitter import TemporalSplitter


class ModelSelector:
    def __init__(
        self,
        n_cv_splits: int = 5,
        tune: bool = False,
        tune_iter: int = 8,
        include_baseline: bool = True,
        freq: str = "D",
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.n_cv_splits = n_cv_splits
        self.tune = tune
        self.tune_iter = tune_iter
        self.include_baseline = include_baseline
        self.freq = freq
        self.random_state = random_state
        self.verbose = verbose
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
        invert_fn: Optional[Callable] = None,
    ) -> "ModelSelector":
        self.candidates = self._build_candidates(X_train, y_train)

        cv_folds = self._safe_cv_folds(X_train)

        self.scoreboard = []
        for model in self.candidates:
            self._log(f"  Training {model.name}...")
            try:
                model.fit(X_train, y_train, cv_splits=cv_folds)

                y_pred_scaled = model.predict(X_test)

                # Exact original-unit predictions via TargetInverter
                y_pred_original = None
                if invert_fn is not None:
                    try:
                        y_pred_original = invert_fn(y_pred_scaled, X_test.index)
                    except Exception:
                        y_pred_original = None

                test_scores = model.score(
                    X_test, y_test,
                    y_test_original=y_test_original,
                    y_pred_original=y_pred_original,
                )

                # Scoreboard shows original-unit metrics when available
                display = model.test_scores_original_ or test_scores
                row = {
                    "model": model.name,
                    **model.cv_scores_,
                    "mae": display.get("mae"),
                    "rmse": display.get("rmse"),
                    "mape": display.get("mape"),
                    "r2": display.get("r2"),
                    "composite_score": model.composite_score,
                }
                self.scoreboard.append(row)
            except Exception as e:
                self._log(f"  {model.name} failed: {e}")

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
            "winner": best_name,
            "scoreboard": self.scoreboard,
            "n_candidates": len(self.candidates),
            "failed_models": len(self.candidates) - len(self.scoreboard),
            "tuned": self.tune,
            "cv_folds_used": len(cv_folds) if cv_folds else 0,
            "baseline_included": self.include_baseline,
        }

        self._log(f"\n  Winner: {best_name} "
                  f"(composite={self.scoreboard[0]['composite_score']:.4f})")
        return self

    def baseline_row(self) -> Optional[Dict]:
        """Scoreboard row of the SeasonalNaive baseline, if it competed."""
        for row in self.scoreboard:
            if row.get("model") == "SeasonalNaive":
                return row
        return None

    def scoreboard_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.scoreboard).set_index("model")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_candidates(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> List[BaseTimeSeriesModel]:
        classes = [RFModel, XGBModel, CatModel]
        if self.tune:
            candidates = [
                self._tune_model(cls, X_train, y_train) for cls in classes
            ]
        else:
            candidates = [cls() for cls in classes]

        if self.include_baseline and any(
            str(c).startswith("lag_") for c in X_train.columns
        ):
            candidates.append(SeasonalNaiveModel(freq=self.freq))
        return candidates

    def _tune_model(self, model_cls, X_train, y_train) -> BaseTimeSeriesModel:
        """
        Random search over the model's PARAM_GRID, validated on the last
        20% of the train split (temporal — no shuffling). Always includes
        the default config, so tuning can never do worse than v0.1.x.
        """
        grid = getattr(model_cls, "PARAM_GRID", {}) or {}
        n = len(X_train)
        val_n = max(5, int(n * 0.2))
        if not grid or n - val_n < 10:
            return model_cls()

        X_tr, X_val = X_train.iloc[: n - val_n], X_train.iloc[n - val_n:]
        y_tr, y_val = y_train.iloc[: n - val_n], y_train.iloc[n - val_n:]

        rng = random.Random(self.random_state)
        configs = [{}]  # default first
        seen = {tuple()}
        for _ in range(self.tune_iter * 3):
            if len(configs) > self.tune_iter:
                break
            cfg = {k: rng.choice(v) for k, v in grid.items()}
            key = tuple(sorted(cfg.items()))
            if key not in seen:
                seen.add(key)
                configs.append(cfg)

        best_cfg, best_mae = {}, float("inf")
        for cfg in configs:
            try:
                m = model_cls(**cfg)
                m.fit(X_tr, y_tr)
                preds = m.predict(X_val)
                mae = float(np.mean(np.abs(y_val.values - preds)))
                if mae < best_mae:
                    best_mae, best_cfg = mae, cfg
            except Exception:
                continue

        self._log(f"  Tuned {model_cls.__name__}: {best_cfg or 'defaults'}")
        return model_cls(**best_cfg)

    def _safe_cv_folds(self, X_train: pd.DataFrame) -> Optional[list]:
        """Walk-forward folds, shrunk (or skipped) for short series."""
        n = len(X_train)
        if n < 15:
            return None
        n_splits = min(self.n_cv_splits, max(2, n // 10))
        splitter = TemporalSplitter(n_splits=n_splits)
        try:
            return splitter.cv_splits(X_train, None)
        except Exception:
            return None

    def _log(self, msg: str):
        if self.verbose:
            print(msg)
