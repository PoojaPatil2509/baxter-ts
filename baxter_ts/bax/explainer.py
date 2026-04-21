"""
BAX Explainer: computes SHAP values on the winning model.
Works with tree-based models (RF, XGB, CatBoost) via TreeExplainer.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class BAXExplainer:
    def __init__(self):
        self._explainer = None
        self.shap_values_: Optional[np.ndarray] = None
        self.feature_importance_: Optional[pd.Series] = None
        self.top_features_: List[str] = []
        self.audit: dict = {}

    def fit(self, model, X_sample: pd.DataFrame) -> "BAXExplainer":
        """Fit SHAP TreeExplainer on a sample of training data."""
        try:
            import shap
            sample = X_sample.dropna()
            if len(sample) > 500:
                sample = sample.sample(500, random_state=42)

            raw_model = model._model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._explainer = shap.TreeExplainer(raw_model)
                self.shap_values_ = self._explainer.shap_values(sample)

            mean_abs = np.abs(self.shap_values_).mean(axis=0)
            self.feature_importance_ = pd.Series(
                mean_abs, index=sample.columns
            ).sort_values(ascending=False)
            self.top_features_ = self.feature_importance_.head(10).index.tolist()
            self.audit["shap_computed"] = True
            self.audit["top_features"] = self.top_features_
        except Exception as e:
            self.audit["shap_computed"] = False
            self.audit["shap_error"] = str(e)
        return self

    def explain_prediction(
        self, model, X_row: pd.DataFrame
    ) -> Dict[str, float]:
        """Returns per-feature SHAP contribution for a single row."""
        if self._explainer is None:
            return {}
        try:
            import shap
            row = X_row.dropna(axis=1).fillna(0)
            vals = self._explainer.shap_values(row)
            return dict(zip(row.columns, vals[0]))
        except Exception:
            return {}
