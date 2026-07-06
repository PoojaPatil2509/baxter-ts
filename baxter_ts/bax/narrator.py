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
        winner_row: Optional[dict] = None,
        baseline_row: Optional[dict] = None,
    ) -> str:
        baseline_line = self._baseline_comparison(
            model_name, winner_row, baseline_row
        )

        if model_name == "SeasonalNaive":
            return self._baseline_winner_narrative(
                target_col, test_scores, original_scores
            )

        if feature_importance is None or len(feature_importance) == 0:
            fallback = self._fallback_narrative(model_name, target_col, test_scores)
            if baseline_line:
                fallback += "\n\n" + baseline_line
            return fallback

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
        if baseline_line:
            lines.append(baseline_line)
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

    def _baseline_comparison(
        self,
        model_name: str,
        winner_row: Optional[dict],
        baseline_row: Optional[dict],
    ) -> str:
        """One-line honesty check: how much did ML beat seasonal-naive by?"""
        if (
            not winner_row or not baseline_row
            or model_name == "SeasonalNaive"
        ):
            return ""
        try:
            winner_mae = float(winner_row.get("mae"))
            baseline_mae = float(baseline_row.get("mae"))
        except (TypeError, ValueError):
            return ""
        if not np.isfinite(winner_mae) or not np.isfinite(baseline_mae) or baseline_mae <= 0:
            return ""
        margin = round((baseline_mae - winner_mae) / baseline_mae * 100, 1)
        if margin >= 1:
            return (
                f"Sanity check: {model_name} beats the seasonal-naive "
                f"baseline by {margin}% (MAE {winner_mae} vs {baseline_mae})."
            )
        return (
            f"Caution: {model_name} improves on the seasonal-naive baseline "
            f"by only {margin}% (MAE {winner_mae} vs {baseline_mae}) — the "
            "series is close to a seasonal random walk, so treat model "
            "gains conservatively."
        )

    def _baseline_winner_narrative(
        self,
        target_col: str,
        test_scores: dict,
        original_scores: Optional[dict],
    ) -> str:
        scores = original_scores or test_scores or {}
        return (
            f"The seasonal-naive baseline — 'this period equals the same "
            f"period one season ago' — outperformed every ML model for "
            f"'{target_col}' (test MAE: {scores.get('mae', 'N/A')}, RMSE: "
            f"{scores.get('rmse', 'N/A')}).\n\n"
            "This usually means the series behaves like a seasonal random "
            "walk: past seasonal values already carry most of the "
            "predictable signal, and additional features add noise rather "
            "than information. The baseline's forecast is used for "
            "predictions. Consider adding informative exogenous columns "
            "(promotions, weather, price) if you need to beat it."
        )

    def _fallback_narrative(
        self, model_name: str, target_col: str, test_scores: dict
    ) -> str:
        return (
            f"{model_name} was selected to forecast '{target_col}'. "
            f"Test MAE: {test_scores.get('mae', 'N/A')}, "
            f"RMSE: {test_scores.get('rmse', 'N/A')}. "
            "SHAP explanation unavailable for this model configuration."
        )