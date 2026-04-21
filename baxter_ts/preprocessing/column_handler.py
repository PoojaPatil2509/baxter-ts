"""
Column handler: prepares multi-column DataFrames before modelling.

Handles:
  - Categorical / object columns  → label-encode or one-hot (auto-chosen)
  - Boolean columns               → cast to int
  - Constant columns              → drop (zero variance, useless)
  - All-NaN columns               → drop
  - ID-like string columns        → drop (too many unique values)
  - Non-target numeric columns    → keep as exogenous features
  - Logs every action for the BAX audit trail
"""

import numpy as np
import pandas as pd
from typing import List, Optional


class ColumnHandler:
    """
    Cleans and encodes all non-target, non-date columns so the pipeline
    receives a fully numeric DataFrame.
    """

    # If a categorical column has ≤ this many unique values → one-hot encode
    # If more → label encode (avoids column explosion on high-cardinality cols)
    OHE_CARDINALITY_LIMIT = 10

    def __init__(self):
        self.dropped_cols:   List[str] = []
        self.encoded_cols:   List[str] = []
        self.ohe_cols:       List[str] = []
        self.label_map:      dict = {}   # col → {value: int} mapping
        self.ohe_dummies:    Optional[pd.Index] = None
        self.audit:          dict = {}

    # ------------------------------------------------------------------
    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str,
        date_col: Optional[str] = None,
    ) -> pd.DataFrame:
        df = df.copy()
        protected = {target_col}
        if date_col and date_col in df.columns:
            protected.add(date_col)

        feature_cols = [c for c in df.columns if c not in protected]

        dropped, encoded_label, encoded_ohe = [], [], []

        for col in feature_cols:
            series = df[col]

            # ── Drop: all NaN ──────────────────────────────────────────
            if series.isna().all():
                df.drop(columns=[col], inplace=True)
                dropped.append(f"{col} (all-NaN)")
                continue

            # ── Drop: constant / zero variance ─────────────────────────
            if series.dropna().nunique() <= 1:
                df.drop(columns=[col], inplace=True)
                dropped.append(f"{col} (constant)")
                continue

            # ── Boolean → int ──────────────────────────────────────────
            if series.dtype == bool or str(series.dtype) == "boolean":
                df[col] = series.astype(int)
                continue

            # ── Object / category → encode ─────────────────────────────
            if (series.dtype == object or str(series.dtype) in ('str', 'string', 'category') or pd.api.types.is_string_dtype(series)):
                n_unique = series.dropna().nunique()

                # Drop: ID-like columns (unique count ≥ 80% of rows)
                if n_unique >= len(df) * 0.8:
                    df.drop(columns=[col], inplace=True)
                    dropped.append(f"{col} (ID-like, {n_unique} unique values)")
                    continue

                if n_unique <= self.OHE_CARDINALITY_LIMIT:
                    # One-hot encode
                    dummies = pd.get_dummies(
                        series.fillna("_missing_"),
                        prefix=col,
                        dtype=int,
                    )
                    df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                    encoded_ohe.append(col)
                else:
                    # Label encode (ordinal integers)
                    categories = series.dropna().unique().tolist()
                    mapping = {v: i for i, v in enumerate(sorted(map(str, categories)))}
                    self.label_map[col] = mapping
                    df[col] = (
                        series.fillna("_missing_")
                        .astype(str)
                        .map(mapping)
                        .fillna(-1)
                        .astype(int)
                    )
                    encoded_label.append(col)
                continue

            # ── Integer-encoded categoricals disguised as int ──────────
            # (no action needed — they pass through as numeric)

        self.dropped_cols  = dropped
        self.encoded_cols  = encoded_label
        self.ohe_cols      = encoded_ohe

        self.audit = {
            "columns_dropped":      dropped,
            "columns_label_encoded": encoded_label,
            "columns_ohe":          encoded_ohe,
            "total_cols_after":     len(df.columns),
        }
        return df

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Apply same encoding to new data (e.g. future forecast rows)."""
        df = df.copy()
        for col, mapping in self.label_map.items():
            if col in df.columns:
                df[col] = (
                    df[col].fillna("_missing_")
                    .astype(str)
                    .map(mapping)
                    .fillna(-1)
                    .astype(int)
                )
        return df
