"""
Anomaly detection on model residuals.
Running on residuals (actual - predicted) catches anomalies the model
didn't predict — far more useful than flagging obvious raw-value spikes.
Three methods: IsolationForest, Z-score on residuals, IQR on residuals.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Optional


class AnomalyDetector:
    SEVERITY_LABELS = {0: "normal", 1: "suspicious", 2: "anomaly"}

    def __init__(
        self,
        method: str = "ensemble",
        contamination: float = 0.05,
        z_threshold: float = 3.0,
        window: int = 30,
    ):
        self.method = method
        self.contamination = contamination
        self.z_threshold = z_threshold
        self.window = window
        self._iso: Optional[IsolationForest] = None
        self.audit: dict = {}

    def fit_predict(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        X: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
            actual, predicted, residual, anomaly_score,
            anomaly_flag (0/1), severity, anomaly_label
        """
        residuals = y_true.values - y_pred
        index = y_true.index

        result = pd.DataFrame(
            {
                "actual": y_true.values,
                "predicted": y_pred,
                "residual": residuals,
            },
            index=index,
        )

        # Method 1: Isolation Forest on residuals
        iso_scores = self._iso_scores(residuals)

        # Method 2: Rolling Z-score on residuals
        zscore_flags = self._rolling_zscore_flags(residuals)

        # Method 3: IQR flags
        iqr_flags = self._iqr_flags(residuals)

        if self.method == "ensemble":
            # Vote: flag if 2+ methods agree
            votes = iso_scores + zscore_flags.astype(int) + iqr_flags.astype(int)
            result["anomaly_score"] = iso_scores.astype(float) + np.abs(residuals) / (np.std(residuals) + 1e-9)
            result["anomaly_flag"] = (votes >= 2).astype(int)
        elif self.method == "isolation_forest":
            result["anomaly_score"] = iso_scores.astype(float)
            result["anomaly_flag"] = iso_scores
        elif self.method == "zscore":
            result["anomaly_score"] = np.abs(residuals) / (np.std(residuals) + 1e-9)
            result["anomaly_flag"] = zscore_flags.astype(int)
        else:
            result["anomaly_score"] = np.abs(residuals) / (np.std(residuals) + 1e-9)
            result["anomaly_flag"] = iqr_flags.astype(int)

        # Severity: 0=normal, 1=suspicious (score > 1.5σ), 2=anomaly (flagged)
        sigma = np.std(residuals) + 1e-9
        result["severity"] = 0
        result.loc[np.abs(residuals) > 1.5 * sigma, "severity"] = 1
        result.loc[result["anomaly_flag"] == 1, "severity"] = 2
        result["severity_label"] = result["severity"].map(self.SEVERITY_LABELS)

        n_anomalies = int(result["anomaly_flag"].sum())
        self.audit = {
            "method": self.method,
            "total_points": len(result),
            "anomalies_found": n_anomalies,
            "anomaly_pct": round(n_anomalies / len(result) * 100, 2),
            "suspicious_count": int((result["severity"] == 1).sum()),
            "residual_mean": round(float(np.mean(residuals)), 4),
            "residual_std": round(float(np.std(residuals)), 4),
        }
        return result

    def _iso_scores(self, residuals: np.ndarray) -> np.ndarray:
        try:
            iso = IsolationForest(
                contamination=self.contamination, random_state=42, n_jobs=-1
            )
            preds = iso.fit_predict(residuals.reshape(-1, 1))
            return (preds == -1).astype(int)
        except Exception:
            return np.zeros(len(residuals), dtype=int)

    def _rolling_zscore_flags(self, residuals: np.ndarray) -> np.ndarray:
        s = pd.Series(residuals)
        roll_mean = s.rolling(self.window, min_periods=3).mean().fillna(s.mean())
        roll_std  = s.rolling(self.window, min_periods=3).std().fillna(s.std())
        z = (s - roll_mean) / (roll_std + 1e-9)
        return (z.abs() > self.z_threshold).values

    def _iqr_flags(self, residuals: np.ndarray) -> np.ndarray:
        q1, q3 = np.percentile(residuals, 25), np.percentile(residuals, 75)
        iqr = q3 - q1
        lower, upper = q1 - 2.5 * iqr, q3 + 2.5 * iqr
        return (residuals < lower) | (residuals > upper)
