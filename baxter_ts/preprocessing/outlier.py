"""Step 4: Outlier detection and treatment — auto-selects method by data shape."""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


class OutlierHandler:
    def __init__(
        self,
        method: str = "auto",
        treatment: str = "cap",
        z_threshold: float = 3.0,
        contamination: float = 0.05,
    ):
        self.method = method
        self.treatment = treatment
        self.z_threshold = z_threshold
        self.contamination = contamination
        self.audit: dict = {}

    def fit_transform(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        df = df.copy()
        series = df[target_col].dropna()

        chosen = self._auto_select(series) if self.method == "auto" else self.method

        if chosen == "zscore":
            mask = self._zscore_mask(series)
        elif chosen == "iqr":
            mask = self._iqr_mask(series)
        elif chosen == "isolation_forest":
            mask = self._isolation_forest_mask(series)
        else:
            mask = pd.Series(False, index=series.index)

        # Align mask to full df index
        full_mask = mask.reindex(df.index, fill_value=False)
        outlier_count = full_mask.sum()

        if self.treatment == "cap":
            q01 = series.quantile(0.01)
            q99 = series.quantile(0.99)
            df[target_col] = df[target_col].clip(lower=q01, upper=q99)
        elif self.treatment == "flag":
            df[f"{target_col}_is_outlier"] = full_mask.astype(int)

        self.audit.update(
            {
                "outlier_method": chosen,
                "outlier_treatment": self.treatment,
                "outliers_found": int(outlier_count),
                "outlier_pct": round(float(outlier_count / len(series) * 100), 2),
            }
        )
        return df

    def _auto_select(self, series: pd.Series) -> str:
        try:
            from scipy.stats import normaltest
            _, p = normaltest(series.dropna())
            if p > 0.05:
                return "zscore"
        except Exception:
            pass
        try:
            skew = float(series.skew())
            if abs(skew) > 1.5:
                return "isolation_forest"
        except Exception:
            pass
        return "iqr"

    def _zscore_mask(self, series: pd.Series) -> pd.Series:
        z = (series - series.mean()) / (series.std() + 1e-9)
        return z.abs() > self.z_threshold

    def _iqr_mask(self, series: pd.Series) -> pd.Series:
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        return (series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)

    def _isolation_forest_mask(self, series: pd.Series) -> pd.Series:
        clf = IsolationForest(contamination=self.contamination, random_state=42)
        vals = series.values.reshape(-1, 1)
        preds = clf.fit_predict(vals)
        return pd.Series(preds == -1, index=series.index)
