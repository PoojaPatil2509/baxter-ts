"""
BAX Narrator: translates SHAP feature importances into plain-English
behavioural explanations. This is the core differentiator of baxter-ts.

Fix v0.1.2:
  - All SHAP percentages rounded to 1dp before string insertion
  - preprocessing_summary reads all audit keys correctly
  - feature count included in narrative
"""

import pandas as pd
import numpy as np
from typing import Optional


class BAXNarrator:
    FEATURE_PHRASES = {
        "lag_":         "recent past values (lag-{n})",
        "roll_mean_":   "rolling average over {n} periods",
        "roll_std_":    "recent volatility ({n}-period std dev)",
        "roll_max_":    "rolling maximum over {n} periods",
        "roll_min_":    "rolling minimum over {n} periods",
        "roll_range_":  "rolling range over {n} periods",
        "seasonal_":    "seasonal pattern",
        "trend_":       "underlying trend",
        "ewm_":         "exponentially weighted recent momentum",
        "dayofweek":    "day-of-week effect",
        "month":        "month-of-year effect",
        "hour":         "hour-of-day effect",
        "is_weekend":   "weekend vs weekday pattern",
        "sin_":         "cyclical seasonality (sine component)",
        "cos_":         "cyclical seasonality (cosine component)",
        "time_idx":     "long-run time trend",
        "pct_change":   "recent rate of change",
        "residual_":    "irregular/noise component",
        "holiday":      "holiday calendar effect",
        "quarter":      "quarterly pattern",
        "is_month":     "month boundary effect",
        "is_year":      "year boundary effect",
        "is_quarter":   "quarter boundary effect",
    }

    def generate(
        self,
        feature_importance: Optional[pd.Series],
        model_name: str,
        target_col: str,
        test_scores: dict,
        preprocessing_audit: dict,
        original_scores: Optional[dict] = None,
    ) -> str:
        if feature_importance is None or len(feature_importance) == 0:
            return self._fallback_narrative(model_name, target_col, test_scores)

        top = feature_importance.head(10)
        total_importance = top.sum()

        # Use original-scale scores for display if available
        display_scores = original_scores if original_scores else test_scores
        mae_display  = display_scores.get("mae",  test_scores.get("mae",  "N/A"))
        rmse_display = display_scores.get("rmse", test_scores.get("rmse", "N/A"))

        lines = []
        lines.append(
            f"The {model_name} model was selected as the best performer for "
            f"predicting '{target_col}', achieving a test MAE of "
            f"{mae_display} and RMSE of {rmse_display}."
        )
        lines.append("")
        lines.append("Key behavioural drivers (BAX analysis):")
        lines.append("")

        cumulative = 0.0
        for rank, (feat, importance) in enumerate(top.items(), start=1):
            # FIX: round to 1dp — prevents "33.29999923706055%" in output
            raw_pct = importance / (total_importance + 1e-9) * 100
            pct = round(float(raw_pct), 1)
            cumulative += pct
            phrase = self._describe_feature(feat)
            lines.append(
                f"  {rank}. {phrase} accounts for {pct}% of prediction influence."
            )
            if cumulative >= 80:
                break

        lines.append("")
        summary = self._preprocessing_summary(preprocessing_audit)
        if summary:
            lines.append(summary)
            lines.append("")
        lines.append(
            "Note: Contributions are computed using SHAP (SHapley Additive "
            "eXplanations), which fairly attributes prediction influence across "
            "all features while respecting feature interactions."
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
                if n:
                    return template.replace("{n}", n)
                return template.replace(" ({n})", "").replace("{n}", "")
        return f"feature '{feature_name}'"

    def _preprocessing_summary(self, audit: dict) -> str:
        parts = []
        v  = audit.get("validator",   {}) or {}
        i  = audit.get("imputer",     {}) or {}
        o  = audit.get("outlier",     {}) or {}
        t  = audit.get("transformer", {}) or {}
        sc = audit.get("scaler",      {}) or {}
        fe = audit.get("feature_eng", {}) or {}

        freq = v.get("inferred_frequency") or v.get("detected_freq")
        if freq:
            parts.append(f"Data frequency detected as '{freq}'")

        miss_pct = i.get("missing_pct", 0)
        if miss_pct and float(miss_pct) > 0:
            parts.append(
                f"{miss_pct}% missing values filled using "
                f"'{i.get('strategy_used', 'auto')}'"
            )

        n_outliers = o.get("outliers_found", 0)
        if n_outliers and int(n_outliers) > 0:
            parts.append(
                f"{n_outliers} outliers ({o.get('outlier_pct', '')}%) "
                f"treated via {o.get('outlier_treatment', 'cap')}"
            )

        n_diffs = t.get("diffs_applied", 0)
        if n_diffs and int(n_diffs) > 0:
            parts.append(f"{n_diffs}-order differencing applied for stationarity")

        if t.get("log_transform_applied"):
            parts.append("log(1+x) transform applied to reduce skewness")

        scaler = sc.get("scaler_used")
        if scaler:
            parts.append(f"{scaler} scaling applied")

        n_features = fe.get("total_features_added")
        if n_features and int(n_features) > 0:
            parts.append(f"{n_features} features engineered automatically")

        if parts:
            return "Preprocessing applied: " + "; ".join(parts) + "."
        return ""

    def _fallback_narrative(
        self, model_name: str, target_col: str, test_scores: dict
    ) -> str:
        return (
            f"{model_name} was selected to forecast '{target_col}'. "
            f"Test MAE: {test_scores.get('mae', 'N/A')}, "
            f"RMSE: {test_scores.get('rmse', 'N/A')}. "
            "SHAP explanation unavailable for this model configuration."
        )