"""
Visualisation engine: all Plotly interactive charts for baxter-ts.
Charts:
  1. Forecast plot (actual vs predicted + confidence interval)
  2. Anomaly overlay (flags on time series)
  3. SHAP feature importance bar chart
  4. Model scoreboard comparison
  5. Residual distribution
  6. Decomposition plot (trend + seasonal + residual)
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict


def _plotly():
    import plotly.graph_objects as go
    import plotly.subplots as sp
    return go, sp


class BAXPlotter:
    COLORS = {
        "actual":    "#378ADD",
        "predicted": "#1D9E75",
        "ci":        "rgba(29,158,117,0.15)",
        "anomaly":   "#D85A30",
        "suspicious":"#EF9F27",
        "grid":      "rgba(0,0,0,0.06)",
        "rf":        "#534AB7",
        "xgb":       "#1D9E75",
        "cat":       "#D85A30",
    }

    def forecast_plot(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        future_dates: Optional[pd.DatetimeIndex] = None,
        future_pred: Optional[np.ndarray] = None,
        ci_lower: Optional[np.ndarray] = None,
        ci_upper: Optional[np.ndarray] = None,
        target_col: str = "target",
    ):
        go, _ = _plotly()
        fig = go.Figure()

        # Actual values
        fig.add_trace(go.Scatter(
            x=y_true.index, y=y_true.values,
            name="Actual", line=dict(color=self.COLORS["actual"], width=2),
            mode="lines",
        ))

        # Test predictions
        pred_index = y_true.index
        fig.add_trace(go.Scatter(
            x=pred_index, y=y_pred,
            name="Predicted", line=dict(color=self.COLORS["predicted"], width=2, dash="dot"),
            mode="lines",
        ))

        # Confidence interval
        if ci_lower is not None and ci_upper is not None:
            fig.add_trace(go.Scatter(
                x=np.concatenate([pred_index, pred_index[::-1]]),
                y=np.concatenate([ci_upper, ci_lower[::-1]]),
                fill="toself", fillcolor=self.COLORS["ci"],
                line=dict(color="rgba(0,0,0,0)"),
                name="95% CI", showlegend=True,
            ))

        # Future forecast
        if future_dates is not None and future_pred is not None:
            fig.add_trace(go.Scatter(
                x=future_dates, y=future_pred,
                name="Forecast", line=dict(color=self.COLORS["predicted"], width=2.5),
                mode="lines",
            ))

        fig.update_layout(
            title=f"Forecast: {target_col}",
            xaxis_title="Date",
            yaxis_title=target_col,
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.05),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_xaxes(showgrid=True, gridcolor=self.COLORS["grid"])
        fig.update_yaxes(showgrid=True, gridcolor=self.COLORS["grid"])
        return fig

    def anomaly_plot(
        self,
        anomaly_df: pd.DataFrame,
        target_col: str = "target",
    ):
        go, _ = _plotly()
        fig = go.Figure()

        # Actual line
        fig.add_trace(go.Scatter(
            x=anomaly_df.index, y=anomaly_df["actual"],
            name="Actual", line=dict(color=self.COLORS["actual"], width=2),
            mode="lines",
        ))

        # Predicted line
        fig.add_trace(go.Scatter(
            x=anomaly_df.index, y=anomaly_df["predicted"],
            name="Predicted", line=dict(color=self.COLORS["predicted"], width=1.5, dash="dot"),
            mode="lines",
        ))

        # Suspicious points
        suspicious = anomaly_df[anomaly_df["severity"] == 1]
        if len(suspicious):
            fig.add_trace(go.Scatter(
                x=suspicious.index, y=suspicious["actual"],
                name="Suspicious", mode="markers",
                marker=dict(color=self.COLORS["suspicious"], size=8, symbol="diamond"),
            ))

        # Anomaly points
        anomalies = anomaly_df[anomaly_df["severity"] == 2]
        if len(anomalies):
            fig.add_trace(go.Scatter(
                x=anomalies.index, y=anomalies["actual"],
                name="Anomaly", mode="markers",
                marker=dict(color=self.COLORS["anomaly"], size=10, symbol="x"),
            ))

        fig.update_layout(
            title=f"Anomaly Detection: {target_col}",
            xaxis_title="Date",
            yaxis_title=target_col,
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.05),
        )
        return fig

    def shap_plot(self, feature_importance: pd.Series, top_n: int = 15):
        go, _ = _plotly()
        top = feature_importance.head(top_n).sort_values()
        total = top.sum() + 1e-9
        pcts = (top / total * 100).round(1)

        fig = go.Figure(go.Bar(
            x=top.values,
            y=top.index,
            orientation="h",
            marker=dict(
                color=top.values,
                colorscale=[[0, "#E6F1FB"], [0.5, "#378ADD"], [1, "#042C53"]],
                showscale=False,
            ),
            text=[f"{p}%" for p in pcts.values],
            textposition="outside",
        ))
        fig.update_layout(
            title="BAX Feature Importance (SHAP)",
            xaxis_title="Mean |SHAP value|",
            yaxis_title="",
            template="plotly_white",
            height=max(350, top_n * 28),
            margin=dict(l=200),
        )
        return fig

    def scoreboard_plot(self, scoreboard: List[Dict]):
        go, sp = _plotly()
        if not scoreboard:
            return go.Figure()

        df = pd.DataFrame(scoreboard)
        metrics = ["mae", "rmse", "mape"]
        model_colors = {
            "RandomForest": self.COLORS["rf"],
            "XGBoost":      self.COLORS["xgb"],
            "CatBoost":     self.COLORS["cat"],
        }

        fig = sp.make_subplots(
            rows=1, cols=3,
            subplot_titles=["MAE (lower=better)", "RMSE (lower=better)", "MAPE % (lower=better)"],
        )
        for col_idx, metric in enumerate(metrics, start=1):
            if metric not in df.columns:
                continue
            for _, row in df.iterrows():
                model = row.get("model", "Unknown")
                fig.add_trace(
                    go.Bar(
                        name=model,
                        x=[model],
                        y=[row.get(metric, 0)],
                        marker_color=model_colors.get(model, "#888"),
                        showlegend=(col_idx == 1),
                    ),
                    row=1, col=col_idx,
                )
        fig.update_layout(
            title="Model Competition Scoreboard",
            template="plotly_white",
            barmode="group",
            height=380,
        )
        return fig

    def residual_plot(self, anomaly_df: pd.DataFrame):
        go, sp = _plotly()
        residuals = anomaly_df["residual"].dropna()

        fig = sp.make_subplots(
            rows=1, cols=2,
            subplot_titles=["Residuals over time", "Residual distribution"],
        )
        # Residuals over time
        colors = anomaly_df["severity"].map({0: self.COLORS["actual"], 1: self.COLORS["suspicious"], 2: self.COLORS["anomaly"]})
        fig.add_trace(
            go.Scatter(
                x=anomaly_df.index,
                y=anomaly_df["residual"],
                mode="markers",
                marker=dict(color=colors.values, size=5),
                name="Residual",
            ),
            row=1, col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)

        # Distribution
        fig.add_trace(
            go.Histogram(x=residuals, nbinsx=40, name="Distribution",
                         marker_color=self.COLORS["actual"]),
            row=1, col=2,
        )
        fig.update_layout(
            title="Residual Analysis",
            template="plotly_white",
            showlegend=False,
            height=350,
        )
        return fig

    def decomposition_plot(self, df: pd.DataFrame, target_col: str):
        go, sp = _plotly()
        needed = ["trend_component", "seasonal_component", "residual_component"]
        if not all(c in df.columns for c in needed):
            return None

        fig = sp.make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            subplot_titles=["Original", "Trend", "Seasonal", "Residual"],
            vertical_spacing=0.06,
        )
        pairs = [
            (target_col, self.COLORS["actual"]),
            ("trend_component", self.COLORS["predicted"]),
            ("seasonal_component", self.COLORS["suspicious"]),
            ("residual_component", self.COLORS["anomaly"]),
        ]
        for i, (col, color) in enumerate(pairs, start=1):
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[col], line=dict(color=color, width=1.5), name=col),
                    row=i, col=1,
                )
        fig.update_layout(
            title="STL Decomposition",
            template="plotly_white",
            height=600,
            showlegend=False,
        )
        return fig

    def save(self, fig, path: str):
        """Save as interactive HTML."""
        fig.write_html(path)

    def save_png(self, fig, path: str):
        """Save as static PNG (requires kaleido)."""
        try:
            fig.write_image(path)
        except Exception as e:
            print(f"PNG export failed (install kaleido): {e}")
