"""
BAXModel: the single public class of baxter-ts.
Orchestrates all 10 preprocessing steps, AutoML competition,
BAX explanation, anomaly detection, visualisation and reporting.

Usage:
    from baxter_ts import BAXModel
    import pandas as pd

    df = pd.read_csv("sales.csv")
    model = BAXModel()
    model.fit(df, target_col="sales", date_col="date")
    model.predict(steps=30)
    model.explain()
    model.anomalies()
    model.visualize()
    model.report("my_report")
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional, List

from baxter_ts.preprocessing.validator   import DatetimeValidator
from baxter_ts.preprocessing.imputer     import TimeSeriesImputer
from baxter_ts.preprocessing.outlier     import OutlierHandler
from baxter_ts.preprocessing.transformer import StationarityTransformer
from baxter_ts.preprocessing.scaler      import TimeSeriesScaler
from baxter_ts.preprocessing.feature_eng import TimeSeriesFeatureEngineer
from baxter_ts.preprocessing.splitter        import TemporalSplitter
from baxter_ts.preprocessing.column_handler   import ColumnHandler
from baxter_ts.models.selector           import ModelSelector
from baxter_ts.bax.explainer             import BAXExplainer
from baxter_ts.bax.narrator              import BAXNarrator
from baxter_ts.anomaly.detector          import AnomalyDetector
from baxter_ts.visualization.plotter     import BAXPlotter
from baxter_ts.report.generator          import ReportGenerator


class BAXModel:
    """
    End-to-end AutoML time series model with behavioural explanation.

    Parameters
    ----------
    test_size : float
        Fraction of data held out for test evaluation (default 0.2).
    n_cv_splits : int
        Number of walk-forward CV folds for model selection (default 5).
    outlier_treatment : str
        'cap' (winsorise) or 'flag' (add boolean column). Default 'cap'.
    anomaly_method : str
        'ensemble' | 'isolation_forest' | 'zscore' | 'iqr'. Default 'ensemble'.
    contamination : float
        Expected anomaly fraction for Isolation Forest (default 0.05).
    verbose : bool
        Print progress to stdout (default True).
    """

    def __init__(
        self,
        test_size: float = 0.2,
        n_cv_splits: int = 5,
        outlier_treatment: str = "cap",
        anomaly_method: str = "ensemble",
        contamination: float = 0.05,
        verbose: bool = True,
    ):
        self.test_size = test_size
        self.n_cv_splits = n_cv_splits
        self.outlier_treatment = outlier_treatment
        self.anomaly_method = anomaly_method
        self.contamination = contamination
        self.verbose = verbose

        # State set after fit()
        self.target_col: Optional[str] = None
        self.date_col: Optional[str] = None
        self._freq: Optional[str] = None
        self._df_processed: Optional[pd.DataFrame] = None
        self._feature_cols: List[str] = []

        # Internal components
        self._validator    = DatetimeValidator()
        self._imputer      = TimeSeriesImputer()
        self._outlier      = OutlierHandler(treatment=outlier_treatment)
        self._transformer  = StationarityTransformer()
        self._scaler       = TimeSeriesScaler()
        self._feat_eng     = TimeSeriesFeatureEngineer()
        self._splitter     = TemporalSplitter(test_size=test_size, n_splits=n_cv_splits)
        self._col_handler = ColumnHandler()
        self._selector:  Optional[ModelSelector]  = None
        self._explainer: Optional[BAXExplainer]   = None
        self._detector:  Optional[AnomalyDetector] = None

        # Results
        self._X_train: Optional[pd.DataFrame] = None
        self._X_test:  Optional[pd.DataFrame] = None
        self._y_train: Optional[pd.Series]    = None
        self._y_test:  Optional[pd.Series]    = None
        self._y_pred_test: Optional[np.ndarray] = None
        self._future_dates: Optional[pd.DatetimeIndex] = None
        self._future_pred:  Optional[np.ndarray]       = None
        self._anomaly_df:   Optional[pd.DataFrame]     = None
        self._anomaly_df_audit: dict = {}
        self._best_scores:  dict = {}
        self._best_scores_original: dict = {}
        self._bax_narrative: Optional[str] = None
        self._preprocessing_audit: dict = {}
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str,
        date_col: Optional[str] = None,
    ) -> "BAXModel":
        """
        Run full pipeline: preprocess → feature engineer → AutoML → BAX explain.

        Parameters
        ----------
        df : pd.DataFrame
            Raw time series data. Must contain a datetime column and target column.
        target_col : str
            Name of the column to forecast.
        date_col : str, optional
            Name of the datetime column. Auto-detected if not provided.
        """
        self.target_col = target_col
        self.date_col = date_col
        self._log("=== baxter-ts: starting pipeline ===")

        # ── Step 1+2: Validate and parse datetime ──────────────────────
        self._log("[1/9] Datetime validation and frequency inference...")
        df = self._validator.fit_transform(df, date_col=date_col, target_col=target_col)
        self._freq = self._validator.detected_freq
        self._log(f"      Detected frequency: {self._freq}  |  rows: {len(df)}")

        # ── Step 1.5: Handle extra columns (categorical encode, drop useless) ──
        self._log("[1.5/9] Column encoding and cleaning...")
        df = self._col_handler.fit_transform(df, target_col=target_col)
        dropped = self._col_handler.audit.get("columns_dropped", [])
        encoded = self._col_handler.audit.get("columns_label_encoded", [])
        ohe     = self._col_handler.audit.get("columns_ohe", [])
        self._log(f"      Dropped: {len(dropped)}  |  Label-encoded: {len(encoded)}  |  OHE: {len(ohe)}")

        # ── Step 3: Impute missing values ──────────────────────────────
        self._log("[2/9] Missing value imputation...")
        df = self._imputer.fit_transform(df, target_col)
        self._log(f"      Strategy: {self._imputer._used_strategy}  |  "
                  f"missing before: {self._imputer.audit['missing_before']}")

        # ── Step 4: Outlier handling ───────────────────────────────────
        self._log("[3/9] Outlier detection and treatment...")
        df = self._outlier.fit_transform(df, target_col)
        self._log(f"      Method: {self._outlier.audit['outlier_method']}  |  "
                  f"found: {self._outlier.audit['outliers_found']}")

        # ── Step 5+6: Stationarity + decomposition ─────────────────────
        self._log("[4/9] Stationarity testing and STL decomposition...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = self._transformer.fit_transform(df, target_col, freq=self._freq)
        self._log(f"      Diffs applied: {self._transformer.n_diffs_applied}  |  "
                  f"STL: {self._transformer.audit.get('stl_decomposition', False)}")

        # ── Step 7: Scaling ────────────────────────────────────────────
        self._log("[5/9] Scaling features...")
        df = self._scaler.fit_transform(df, target_col)
        self._log(f"      Scaler: {self._scaler._chosen}")

        # ── Step 8: Feature engineering ────────────────────────────────
        self._log("[6/9] Feature engineering (lags, rolling, Fourier, calendar)...")
        df = self._feat_eng.fit_transform(df, target_col, freq=self._freq)
        self._log(f"      Features added: {len(self._feat_eng.feature_names_)}")

        # Drop rows with NaN from lagging (keep at least 70% of data)
        df = df.dropna(subset=[target_col])
        df = df.fillna(0)   # remaining NaN in feature cols → 0

        self._df_processed = df

        # ── Step 9: Temporal split ─────────────────────────────────────
        self._log("[7/9] Temporal train/test split...")
        self._X_train, self._X_test, self._y_train, self._y_test = \
            self._splitter.split(df, target_col)
        self._feature_cols = self._X_train.columns.tolist()
        self._log(f"      Train: {len(self._X_train)} rows  |  Test: {len(self._X_test)} rows")

        # ── Inverse-transform y_test for original-scale metrics ──────
        try:
            y_test_original = self._scaler.inverse_transform_target(
                self._y_test.values, target_col, self._df_processed
            )
        except Exception:
            y_test_original = None

        # ── AutoML model competition ───────────────────────────────────
        self._log("[8/9] AutoML model competition (RF vs XGBoost vs CatBoost)...")
        self._selector = ModelSelector(n_cv_splits=self.n_cv_splits)
        self._selector.fit(
            self._X_train, self._y_train,
            self._X_test,  self._y_test,
            y_test_original=y_test_original,
        )
        self._best_scores = self._selector.best_model.test_scores_
        self._best_scores_original = (
            self._selector.best_model.test_scores_original_ or {}
        )

        # Generate test predictions
        self._y_pred_test = self._selector.best_model.predict(self._X_test)

        try:
            self._y_pred_test_original = self._scaler.inverse_transform_target(
                self._y_pred_test, target_col, self._df_processed
            )
        except Exception:
            self._y_pred_test_original = self._y_pred_test

        # ── BAX explanation ────────────────────────────────────────────
        self._log("[9/9] Computing BAX explanation (SHAP)...")
        self._explainer = BAXExplainer()
        self._explainer.fit(self._selector.best_model, self._X_train)

        narrator = BAXNarrator()
        self._preprocessing_audit = {
            "column_handler": self._col_handler.audit,
            "validator":      self._validator.audit,
            "imputer":        self._imputer.audit,
            "outlier":        self._outlier.audit,
            "transformer":    self._transformer.audit,
            "scaler":         self._scaler.audit,
            "feature_eng":    self._feat_eng.audit,
            "splitter":       self._splitter.audit,
        }
        self._bax_narrative = narrator.generate(
            feature_importance=self._explainer.feature_importance_,
            model_name=self._selector.best_model.name,
            target_col=target_col,
            test_scores=self._best_scores,
            preprocessing_audit=self._preprocessing_audit,
            original_scores=self._best_scores_original,
        )

        display_scores = self._best_scores_original or self._best_scores

        self._is_fitted = True
        self._log("\n=== Pipeline complete ===")
        self._log(f"    Winner : {self._selector.best_model.name}")
        self._log(f"    MAE    : {display_scores.get('mae')}")
        self._log(f"    RMSE   : {display_scores.get('rmse')}")
        self._log(f"    MAPE   : {display_scores.get('mape')}%")
        self._log(f"    R²     : {display_scores.get('r2')}")
        return self

    def predict(self, steps: int = 30) -> pd.DataFrame:
        """
        Generate future forecast for `steps` time periods ahead.
        Uses a sliding-window direct forecast: replicate the last known
        feature row, advance calendar/Fourier features step-by-step,
        and recycle predicted values into lag slots.

        Returns a DataFrame with columns: date, forecast.
        """
        self._check_fitted()
        self._log(f"\nGenerating {steps}-step forecast...")

        last_date = self._df_processed.index[-1]
        freq = self._freq or "D"

        try:
            future_dates = pd.date_range(start=last_date, periods=steps + 1, freq=freq)[1:]
        except Exception:
            future_dates = pd.date_range(start=last_date, periods=steps + 1, freq="D")[1:]

        # ── Build a sliding window of the last max_lag rows of target values
        # so we can fill lag_N and rolling features properly each step
        max_lag = max(self._feat_eng.lags) if self._feat_eng.lags else 30
        window_size = max_lag + max(self._feat_eng.rolling_windows or [30])

        # Pull the last window of processed target values (scaled space)
        target_series = self._df_processed[self.target_col].dropna().values
        history = list(target_series[-window_size:])

        # Take the last processed feature row as a template for non-time features
        last_feat_row = self._df_processed[self._feature_cols].iloc[-1].copy()

        future_preds = []
        lags = self._feat_eng.lags or [1, 7, 14, 21, 30]
        windows = self._feat_eng.rolling_windows or [7, 14, 30]
        period_map = {"min": 1440, "h": 24, "D": 365.25, "W": 52, "MS": 12, "YS": 1, "Q": 4}
        period = period_map.get(freq, 365.25)
        total_len = len(self._df_processed)

        for step_i, step_date in enumerate(future_dates):
            row = last_feat_row.copy()

            # 1. Update lag features from rolling history
            for lag in lags:
                col = f"lag_{lag}"
                if col in row.index and len(history) >= lag:
                    row[col] = history[-lag]

            # 2. Update rolling stat features
            for win in windows:
                hist_win = history[-win:] if len(history) >= win else history
                hist_arr = np.array(hist_win)
                for stat, col in [
                    ("mean", f"roll_mean_{win}"),
                    ("std",  f"roll_std_{win}"),
                    ("min",  f"roll_min_{win}"),
                    ("max",  f"roll_max_{win}"),
                ]:
                    if col in row.index:
                        if stat == "mean":  row[col] = float(np.mean(hist_arr))
                        elif stat == "std": row[col] = float(np.std(hist_arr)) if len(hist_arr) > 1 else 0.0
                        elif stat == "min": row[col] = float(np.min(hist_arr))
                        elif stat == "max": row[col] = float(np.max(hist_arr))
                        roll_range_col = f"roll_range_{win}"
                        if roll_range_col in row.index:
                            row[roll_range_col] = float(np.max(hist_arr) - np.min(hist_arr))

            # 3. Update EWM features (approximate via last value)
            for alpha_str, col in [("02", "ewm_02"), ("05", "ewm_05"), ("08", "ewm_08")]:
                if col in row.index and len(history) >= 2:
                    alpha = float("0." + alpha_str)
                    ewm_val = history[-1]
                    for v in reversed(history[-10:]):
                        ewm_val = alpha * v + (1 - alpha) * ewm_val
                    row[col] = ewm_val

            # 4. Momentum
            if "pct_change_1" in row.index and len(history) >= 2:
                denom = history[-2]
                row["pct_change_1"] = (history[-1] - denom) / (abs(denom) + 1e-9)
            if "pct_change_7" in row.index and len(history) >= 8:
                denom = history[-8]
                row["pct_change_7"] = (history[-1] - denom) / (abs(denom) + 1e-9)

            # 5. Advance time index
            t = total_len + step_i
            if "time_idx" in row.index:
                row["time_idx"] = t

            # 6. Update calendar features
            if "dayofweek" in row.index:   row["dayofweek"]  = step_date.dayofweek
            if "month" in row.index:       row["month"]      = step_date.month
            if "quarter" in row.index:     row["quarter"]    = step_date.quarter
            if "dayofyear" in row.index:   row["dayofyear"]  = step_date.dayofyear
            if "is_weekend" in row.index:  row["is_weekend"] = int(step_date.dayofweek >= 5)
            if "hour" in row.index:        row["hour"]       = step_date.hour
            if "is_month_start" in row.index: row["is_month_start"] = int(step_date.is_month_start)
            if "is_month_end" in row.index:   row["is_month_end"]   = int(step_date.is_month_end)
            if "is_quarter_end" in row.index: row["is_quarter_end"] = int(step_date.is_quarter_end)

            # 7. Advance Fourier terms
            fourier_order = self._feat_eng.fourier_order if self._feat_eng else 3
            for k in range(1, fourier_order + 1):
                if f"sin_k{k}" in row.index:
                    row[f"sin_k{k}"] = np.sin(2 * np.pi * k * t / period)
                if f"cos_k{k}" in row.index:
                    row[f"cos_k{k}"] = np.cos(2 * np.pi * k * t / period)

            # 8. Predict
            X_row = pd.DataFrame([row[self._feature_cols]])
            X_row = X_row.fillna(0)
            pred_val = float(self._selector.best_model.predict(X_row)[0])
            future_preds.append(pred_val)

            # 9. Append to history for next step
            history.append(pred_val)

        self._future_dates = future_dates
        self._future_pred = np.array(future_preds)

        result = pd.DataFrame({"date": future_dates, "forecast": future_preds})
        result.set_index("date", inplace=True)
        self._log(f"  Forecast generated for {steps} steps.")
        return result

    def anomalies(self) -> pd.DataFrame:
        """
        Run anomaly detection on the test set residuals.
        Returns DataFrame with: actual, predicted, residual, anomaly_flag, severity_label.
        """
        self._check_fitted()
        self._log("\nRunning anomaly detection...")
        self._detector = AnomalyDetector(
            method=self.anomaly_method,
            contamination=self.contamination,
        )
        self._anomaly_df = self._detector.fit_predict(self._y_test, self._y_pred_test)
        self._anomaly_df_audit = self._detector.audit
        self._log(f"  Anomalies found: {self._detector.audit['anomalies_found']} "
                  f"({self._detector.audit['anomaly_pct']}%)")
        return self._anomaly_df

    def explain(self) -> str:
        """Print and return the BAX behavioural narrative."""
        self._check_fitted()
        if self._bax_narrative:
            print("\n" + "=" * 60)
            print("BAX BEHAVIOURAL EXPLANATION")
            print("=" * 60)
            print(self._bax_narrative)
            print("=" * 60 + "\n")
        return self._bax_narrative or ""

    def scoreboard(self) -> pd.DataFrame:
        """Return the model competition scoreboard as a DataFrame."""
        self._check_fitted()
        return self._selector.scoreboard_df()

    def visualize(self, show: bool = True) -> dict:
        """
        Generate all interactive Plotly charts.
        Returns a dict of figure objects.
        Set show=False to suppress auto-display (useful in scripts).
        """
        self._check_fitted()
        plotter = BAXPlotter()
        figs = {}

        if self._y_test is not None and self._y_pred_test is not None:
            figs["forecast"] = plotter.forecast_plot(
                self._y_test, self._y_pred_test,
                future_dates=self._future_dates,
                future_pred=self._future_pred,
                target_col=self.target_col,
            )

        if self._anomaly_df is None:
            self.anomalies()

        if self._anomaly_df is not None:
            figs["anomaly"]  = plotter.anomaly_plot(self._anomaly_df, self.target_col)
            figs["residuals"]= plotter.residual_plot(self._anomaly_df)

        if self._explainer and self._explainer.feature_importance_ is not None:
            figs["shap"] = plotter.shap_plot(self._explainer.feature_importance_)

        if self._selector:
            figs["scoreboard"] = plotter.scoreboard_plot(
                self._selector.audit.get("scoreboard", [])
            )

        if self._df_processed is not None:
            fig_d = plotter.decomposition_plot(self._df_processed, self.target_col)
            if fig_d:
                figs["decomposition"] = fig_d

        if show:
            for name, fig in figs.items():
                fig.show()

        return figs

    def report(self, output_path: str = "bax_report") -> str:
        """
        Export full HTML report with all charts, metrics, narrative and audit trail.

        Parameters
        ----------
        output_path : str
            File path (without extension). Saves as output_path.html.

        Returns the saved file path.
        """
        self._check_fitted()
        if self._anomaly_df is None:
            self.anomalies()
        gen = ReportGenerator()
        return gen.generate(self, output_path)

    def summary(self) -> dict:
        """Return a flat dict of all key results — useful for programmatic access."""
        self._check_fitted()
        return {
            "target_col": self.target_col,
            "frequency": self._freq,
            "best_model": self._selector.best_model.name if self._selector else None,
            "test_mae":  self._best_scores.get("mae"),
            "test_rmse": self._best_scores.get("rmse"),
            "test_mape": self._best_scores.get("mape"),
            "test_r2":   self._best_scores.get("r2"),
            "anomalies_found": self._anomaly_df_audit.get("anomalies_found"),
            "shap_top_features": self._explainer.top_features_ if self._explainer else [],
            "train_rows": self._splitter.audit.get("train_size"),
            "test_rows":  self._splitter.audit.get("test_size"),
        }

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before using this method.")

    def _log(self, msg: str):
        if self.verbose:
            print(msg)
