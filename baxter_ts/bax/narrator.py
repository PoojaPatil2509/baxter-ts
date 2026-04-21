"""
BAX Narrator: translates SHAP feature importances into plain-English
behavioural explanations. This is the core differentiator of baxter-ts.
"""

import pandas as pd
import numpy as np
from typing import Optional


class BAXNarrator:
    FEATURE_PHRASES = {
        "lag_":         "recent past values (lag-{n})",
        "roll_mean_":   "rolling average over {n} periods",
        "roll_std_":    "recent volatility ({n}-period std dev)",
        "seasonal_":    "seasonal pattern",
        "trend_":       "underlying trend",
        "ewm_":         "exponentially weighted recent momentum",
        "dayofweek":    "day-of-week effect",
        "month":        "month-of-year effect",
        "hour":         "hour-of-day effect",
        "is_weekend":   "weekend vs weekday pattern",
        "fourier":      "cyclical seasonality (Fourier term)",
        "sin_":         "cyclical seasonality (sine component)",
        "cos_":         "cyclical seasonality (cosine component)",
        "time_idx":     "long-run time trend",
        "pct_change":   "recent rate of change",
        "residual_":    "irregular/noise component",
        "holiday":      "holiday calendar effect",
        "quarter":      "quarterly pattern",
    }

    def generate(
        self,
        feature_importance: Optional[pd.Series],
        model_name: str,
        target_col: str,
        test_scores: dict,
        preprocessing_audit: dict,
    ) -> str:
        if feature_importance is None or len(feature_importance) == 0:
            return self._fallback_narrative(model_name, target_col, test_scores)

        top = feature_importance.head(10)
        total_importance = top.sum()

        lines = []
        lines.append(
            f"The {model_name} model was selected as the best performer for "
            f"predicting '{target_col}', achieving a test MAE of "
            f"{test_scores.get('mae', 'N/A')} and RMSE of "
            f"{test_scores.get('rmse', 'N/A')}."
        )
        lines.append("")
        lines.append("Key behavioural drivers (BAX analysis):")
        lines.append("")

        cumulative = 0.0
        for rank, (feat, importance) in enumerate(top.items(), start=1):
            pct = round(importance / (total_importance + 1e-9) * 100, 1)
            cumulative += pct
            phrase = self._describe_feature(feat)
            direction = ""
            lines.append(
                f"  {rank}. {phrase} accounts for {pct}% of prediction influence{direction}."
            )
            if cumulative >= 80:
                break

        lines.append("")
        lines.append(self._preprocessing_summary(preprocessing_audit))
        lines.append("")
        lines.append(
            "Note: Contributions are computed using SHAP (SHapley Additive exPlanations), "
            "which fairly attributes prediction influence across all features while "
            "respecting feature interactions."
        )
        return "\n".join(lines)

    def _describe_feature(self, feature_name: str) -> str:
        for prefix, template in self.FEATURE_PHRASES.items():
            if prefix in feature_name:
                n = ""
                parts = feature_name.split("_")
                for part in parts:
                    if part.isdigit():
                        n = part
                        break
                return template.replace("{n}", n) if n else template.replace(" ({n})", "").replace("{n}", "")
        return f"feature '{feature_name}'"

    def _preprocessing_summary(self, audit: dict) -> str:
        parts = []
        v = audit.get("validator", {})
        i = audit.get("imputer", {})
        o = audit.get("outlier", {})
        t = audit.get("transformer", {})
        sc = audit.get("scaler", {})

        if v.get("inferred_frequency"):
            parts.append(f"Data frequency detected as '{v['inferred_frequency']}'")
        if i.get("missing_pct", 0) > 0:
            parts.append(
                f"{i['missing_pct']}% missing values filled using '{i.get('strategy_used', 'auto')}'"
            )
        if o.get("outliers_found", 0) > 0:
            parts.append(
                f"{o['outliers_found']} outliers ({o.get('outlier_pct', '')}%) "
                f"treated via {o.get('outlier_treatment', 'cap')}"
            )
        if t.get("diffs_applied", 0) > 0:
            parts.append(f"{t['diffs_applied']}-order differencing applied for stationarity")
        if t.get("log_transform_applied"):
            parts.append("log(1+x) transform applied to reduce skewness")
        if sc.get("scaler_used"):
            parts.append(f"{sc['scaler_used']} scaling applied")

        if parts:
            return "Preprocessing applied: " + "; ".join(parts) + "."
        return ""

    def _fallback_narrative(self, model_name: str, target_col: str, test_scores: dict) -> str:
        return (
            f"{model_name} was selected to forecast '{target_col}'. "
            f"Test MAE: {test_scores.get('mae', 'N/A')}, "
            f"RMSE: {test_scores.get('rmse', 'N/A')}. "
            "SHAP explanation unavailable for this model configuration."
        )
