"""
TargetInverter: exact inverse of the target transformation chain.

fit() applies these transforms to the target, in order:

    1. log1p          (StationarityTransformer, when skewed + non-stationary)
    2. diff(d)        (StationarityTransformer, a single d-lag difference)
    3. scaling        (TimeSeriesScaler, MinMax/Standard/Robust)

Model predictions therefore live in scaled-differenced-log space. v0.1.x
returned them as-is from predict() (a user forecasting prices near 150 got
numbers like 0.02) and approximated display metrics with a std-ratio hack.
This class walks the chain backwards instead, so both forecasts and metrics
are exact in the units the data arrived in.

Un-differencing needs the value d steps earlier in log space:
    y_t = diff_t + y_{t-d}

  - In-sample (test set): true previous values are known, so every point is
    anchored to actuals (teacher forcing).
  - Future: anchored to the tail of the observed series, then recursively to
    the model's own reconstructed values.
"""

import numpy as np
import pandas as pd


class TargetInverter:
    def __init__(self, scaler, transformer, target_col: str, raw_target: pd.Series):
        """
        Parameters
        ----------
        scaler : TimeSeriesScaler
            Fitted scaler (step 7).
        transformer : StationarityTransformer
            Fitted transformer (steps 5+6) — provides log/diff state.
        target_col : str
            Target column name.
        raw_target : pd.Series
            Target after imputation + outlier handling but BEFORE
            log/diff/scaling — i.e. the series in original units.
        """
        self.scaler = scaler
        self.transformer = transformer
        self.target_col = target_col
        self.log_applied = bool(getattr(transformer, "log_applied", False))
        self.diff_lag = int(getattr(transformer, "n_diffs_applied", 0) or 0)

        raw = raw_target.astype(float)
        self.raw_target = raw
        # The series as it looked between log and diff — the space in which
        # un-differencing anchors live.
        self._log_series = np.log1p(raw) if self.log_applied else raw

    # ------------------------------------------------------------------

    def invert_insample(self, pred_scaled: np.ndarray, index: pd.Index) -> np.ndarray:
        """
        Invert test-set predictions to original units, anchoring each
        un-differencing step to the ACTUAL value d steps earlier.
        """
        vals = self._unscale(pred_scaled)
        if self.diff_lag > 0:
            base = self._log_series
            pos = base.index.get_indexer(index)
            out = np.empty(len(vals), dtype=float)
            for i, (p, v) in enumerate(zip(pos, vals)):
                if p >= self.diff_lag:
                    out[i] = v + base.iloc[p - self.diff_lag]
                elif p >= 0:
                    out[i] = v + base.iloc[0]
                else:  # index not found — should not happen, stay safe
                    out[i] = v + base.iloc[-self.diff_lag]
            vals = out
        if self.log_applied:
            vals = np.expm1(np.clip(vals, -700, 700))
        return vals

    def invert_future(self, pred_scaled: np.ndarray) -> np.ndarray:
        """
        Invert future forecasts to original units. The first d steps anchor
        to the observed tail of the series; later steps anchor recursively
        to already-reconstructed forecasts.
        """
        vals = self._unscale(pred_scaled)
        if self.diff_lag > 0:
            tail = list(self._log_series.iloc[-self.diff_lag:].values)
            out = []
            for v in vals:
                y = v + tail[-self.diff_lag]
                out.append(y)
                tail.append(y)
            vals = np.asarray(out, dtype=float)
        if self.log_applied:
            vals = np.expm1(np.clip(vals, -700, 700))
        return vals

    def actuals_for(self, index: pd.Index) -> np.ndarray:
        """Original-unit actual values aligned to `index` (NaN where missing)."""
        return self.raw_target.reindex(index).values.astype(float)

    # ------------------------------------------------------------------

    def _unscale(self, values) -> np.ndarray:
        arr = np.asarray(values, dtype=float).ravel()
        if self.scaler is None:
            return arr
        try:
            return np.asarray(
                self.scaler.inverse_transform_target(arr, self.target_col, None),
                dtype=float,
            ).ravel()
        except Exception:
            return arr
