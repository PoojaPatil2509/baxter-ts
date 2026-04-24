"""
Report generator: exports a 100% self-contained HTML file.

THE PLOTLY FIX (permanent, production-grade):
  Plotly JS is read directly from the installed plotly Python package
  (plotly/package_data/plotly.min.js) and embedded as a single inline
  <script> block in the HTML <head>.

  Result:
    - Zero CDN calls, zero network requests
    - Works in sandboxed iframes (Claude, VS Code preview)
    - Works fully offline
    - Works in production with no external dependencies
    - One copy of Plotly loaded before any chart code runs, no race condition
"""

import os
import datetime
import pandas as pd
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from baxter_ts.core import BAXModel


def _get_plotly_js_tag() -> str:
    """
    Returns a <script> block with the full Plotly JS bundle embedded inline.
    Source: plotly/package_data/plotly.min.js inside the installed package.
    This is identical to what plotly does when you call
    fig.to_html(include_plotlyjs='inline'), but done once for the whole report.
    """
    try:
        import plotly as _p
        js_path = os.path.join(
            os.path.dirname(_p.__file__), "package_data", "plotly.min.js"
        )
        if os.path.exists(js_path):
            with open(js_path, "r", encoding="utf-8") as fh:
                js = fh.read()
            return '<script type="text/javascript">' + js + "</script>"
    except Exception:
        pass
    raise RuntimeError(
        "plotly.min.js not found. Run: pip install --upgrade plotly"
    )


class ReportGenerator:
    def __init__(self):
        try:
            import baxter_ts as _bt
            self.version = _bt.__version__
        except Exception:
            self.version = "0.1.3"

    def generate(self, model: "BAXModel", output_path: str = "bax_report") -> str:
        from baxter_ts.visualization.plotter import BAXPlotter

        plotter = BAXPlotter()

        def _chart(fig) -> str:
            """
            Embed chart HTML with include_plotlyjs=False.
            Plotly is already in <head> — no per-chart CDN or duplicate JS.
            """
            if fig is None:
                return "<p style='color:#999;font-size:13px'>Chart unavailable.</p>"
            try:
                return fig.to_html(
                    full_html=False,
                    include_plotlyjs=False,
                    config={"responsive": True},
                )
            except Exception as exc:
                return f"<p style='color:#A32D2D'>Chart error: {exc}</p>"

        # Build figures
        f_forecast = f_anomaly = f_shap = f_score = f_resid = f_decomp = None

        if model._y_test is not None and model._y_pred_test is not None:
            f_forecast = plotter.forecast_plot(
                model._y_test, model._y_pred_test,
                future_dates=model._future_dates,
                future_pred=model._future_pred,
                target_col=model.target_col,
            )
        if model._anomaly_df is not None:
            f_anomaly = plotter.anomaly_plot(model._anomaly_df, model.target_col)
            f_resid   = plotter.residual_plot(model._anomaly_df)
        if model._explainer and model._explainer.feature_importance_ is not None:
            f_shap = plotter.shap_plot(model._explainer.feature_importance_)
        if model._selector:
            f_score = plotter.scoreboard_plot(
                model._selector.audit.get("scoreboard", [])
            )
        if model._df_processed is not None:
            f_decomp = plotter.decomposition_plot(
                model._df_processed, model.target_col
            )

        best_name  = model._selector.best_model.name if model._selector else "N/A"
        scores     = (model._best_scores_original or model._best_scores) if hasattr(model, "_best_scores_original") else (model._best_scores or {})
        anom_audit = model._anomaly_df_audit or {}
        now        = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        decomp_block = ""
        if f_decomp:
            decomp_block = (
                "<h2>STL decomposition</h2>"
                f"<div class='card chart-wrap'>{_chart(f_decomp)}</div>"
            )

        # Assemble body HTML
        body_parts = [
            f"<h1>BAX Analysis Report</h1>",
            f"<p class='subtitle'>Target: <strong>{model.target_col}</strong>"
            f" &nbsp;|&nbsp; Model: <span class='badge badge-green'>{best_name}</span>"
            f" &nbsp;|&nbsp; Generated: {now}</p>",

            "<h2>Model performance</h2>",
            self._perf_card(scores),

            "<h2>AutoML competition scoreboard</h2>",
            self._scoreboard_card(model, best_name),

            "<h2>BAX behavioural explanation</h2>",
            self._narrative_card(model),

            "<h2>Anomaly detection summary</h2>",
            self._anomaly_card(model, anom_audit),

            "<h2>Preprocessing audit trail</h2>",
            self._audit_card(model),

            "<h2>Forecast chart</h2>",
            f"<div class='card chart-wrap'>{_chart(f_forecast)}</div>",

            "<h2>Anomaly overlay</h2>",
            f"<div class='card chart-wrap'>{_chart(f_anomaly)}</div>",

            "<h2>Feature importance (SHAP)</h2>",
            f"<div class='card chart-wrap'>{_chart(f_shap)}</div>",

            "<h2>Model scoreboard chart</h2>",
            f"<div class='card chart-wrap'>{_chart(f_score)}</div>",

            "<h2>Residual analysis</h2>",
            f"<div class='card chart-wrap'>{_chart(f_resid)}</div>",

            decomp_block,
        ]
        body_html = "\n".join(body_parts)

        # Build final HTML
        html = self._html_page(
            title=f"BAX Report — {model.target_col}",
            plotly_js_tag=_get_plotly_js_tag(),
            body=body_html,
            version=self.version,
        )

        if not output_path.endswith(".html"):
            output_path += ".html"
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        print(f"  Report saved: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _perf_card(self, scores: dict) -> str:
        def metric(label, key, sub, color=""):
            val = scores.get(key, "N/A")
            suf = "%" if key == "mape" else ""
            style = f" style='color:{color}'" if color else ""
            return (
                f"<div class='metric'>"
                f"<div class='metric-label'>{label}</div>"
                f"<div class='metric-value'{style}>{val}{suf}</div>"
                f"<div class='metric-sub'>{sub}</div></div>"
            )
        return (
            "<div class='card'><div class='metric-grid'>"
            + metric("MAE",  "mae",  "Mean absolute error")
            + metric("RMSE", "rmse", "Root mean sq error")
            + metric("MAPE", "mape", "Mean abs pct error")
            + metric("R\u00b2",  "r2",   "Explained variance")
            + "</div></div>"
        )

    def _scoreboard_card(self, model: "BAXModel", best_name: str) -> str:
        rows_html = ""
        for row in (model._selector.audit.get("scoreboard", []) if model._selector else []):
            win   = row.get("model") == best_name
            cls   = "winner-row" if win else ""
            badge = "<span class='badge badge-green'>Winner</span>" if win else ""
            rows_html += (
                f"<tr class='{cls}'>"
                f"<td>{row.get('model','')}{badge}</td>"
                f"<td>{row.get('mae','')}</td>"
                f"<td>{row.get('rmse','')}</td>"
                f"<td>{row.get('mape','')}</td>"
                f"<td>{row.get('r2','')}</td>"
                f"<td>{row.get('composite_score','')}</td>"
                f"<td>{'Winner' if win else 'Candidate'}</td></tr>"
            )
        return (
            "<div class='card'><table>"
            "<thead><tr>"
            "<th>Model</th><th>MAE</th><th>RMSE</th><th>MAPE%</th>"
            "<th>R\u00b2</th><th>Composite</th><th>Status</th>"
            "</tr></thead>"
            f"<tbody>{rows_html}</tbody></table></div>"
        )

    def _narrative_card(self, model: "BAXModel") -> str:
        text = (model._bax_narrative or "Explanation not available.").replace(
            "\n", "<br>"
        )
        return f"<div class='card'><div class='narrative'>{text}</div></div>"

    def _anomaly_card(self, model: "BAXModel", anom_audit: dict) -> str:
        def m(label, val, sub="", color=""):
            st = f" style='color:{color}'" if color else ""
            return (
                f"<div class='metric'>"
                f"<div class='metric-label'>{label}</div>"
                f"<div class='metric-value'{st}>{val}</div>"
                f"<div class='metric-sub'>{sub}</div></div>"
            )
        grid = (
            "<div class='metric-grid'>"
            + m("Total points",  anom_audit.get("total_points", "N/A"))
            + m("Anomalies",     anom_audit.get("anomalies_found", "N/A"),
                f"{anom_audit.get('anomaly_pct', 'N/A')}% of series", "#A32D2D")
            + m("Suspicious",    anom_audit.get("suspicious_count", "N/A"),
                "borderline points", "#854F0B")
            + m("Method",        anom_audit.get("method", "N/A"))
            + "</div>"
        )
        table_html = ""
        if model._anomaly_df is not None:
            top = model._anomaly_df[model._anomaly_df["anomaly_flag"] == 1].head(10)
            if len(top):
                trows = ""
                for ts, r in top.iterrows():
                    import math
                    sev = r["severity_label"]
                    av  = round(float(r["actual"]),    4) if not (isinstance(r["actual"],    float) and math.isnan(r["actual"]))    else "—"
                    pv  = round(float(r["predicted"]), 4) if not (isinstance(r["predicted"], float) and math.isnan(r["predicted"])) else "—"
                    rv  = round(float(r["residual"]),  4) if not (isinstance(r["residual"],  float) and math.isnan(r["residual"]))  else "—"
                    trows += (
                        f"<tr><td>{ts}</td><td>{av}</td><td>{pv}</td>"
                        f"<td>{rv}</td>"
                        f"<td class='anomaly-{sev}'>{sev}</td></tr>"
                    )
                table_html = (
                    "<h3 style='margin-top:20px'>Top anomalies detected</h3>"
                    "<table style='margin-top:8px'>"
                    "<thead><tr><th>Timestamp</th><th>Actual</th>"
                    "<th>Predicted</th><th>Residual</th><th>Severity</th></tr></thead>"
                    f"<tbody>{trows}</tbody></table>"
                )
        return f"<div class='card'>{grid}{table_html}</div>"

    def _audit_card(self, model: "BAXModel") -> str:
        flat = {}
        for section, vals in model._preprocessing_audit.items():
            if isinstance(vals, dict):
                for k, v in vals.items():
                    flat[f"{section}.{k}"] = v
            else:
                flat[section] = vals
        items = ""
        count = 0
        for k, v in flat.items():
            if count >= 50:
                break
            # Skip genuinely empty values.
            # IMPORTANT: check isinstance(v, bool) FIRST — in Python, bool is
            # a subclass of int, so isinstance(False, (int,float)) is True and
            # False == 0 is True. The old filter wrongly skipped
            # was_stationary=False by treating it as numeric zero.
            if v is None or v == "" or v == [] or v == {}:
                continue
            if not isinstance(v, bool) and isinstance(v, (int, float)) and v == 0:
                keep_keys = ("found", "before", "applied", "size",
                             "rows", "pct", "after", "total")
                if not any(kw in k for kw in keep_keys):
                    continue
            if isinstance(v, list):
                v_str = ", ".join(str(i) for i in v[:5]) if v else ""
                if not v_str:
                    continue
                v = v_str
            display_val = str(v)
            items += (
                f"<div class='audit-item'>"
                f"<div class='audit-key'>{k}</div>"
                f"<div class='audit-val'>{display_val}</div></div>"
            )
            count += 1
        return f"<div class='card'><div class='audit-grid'>{items}</div></div>"
    # ------------------------------------------------------------------
    # HTML page wrapper
    # ------------------------------------------------------------------

    @staticmethod
    def _html_page(title: str, plotly_js_tag: str, body: str, version: str) -> str:
        css = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:#f8f8f6;color:#1a1a18;line-height:1.7;font-size:15px}
.page{max-width:1100px;margin:0 auto;padding:40px 24px}
h1{font-size:28px;font-weight:500;margin-bottom:4px}
h2{font-size:18px;font-weight:500;margin:32px 0 12px;color:#1a1a18;
   border-bottom:1px solid #e0e0da;padding-bottom:6px}
h3{font-size:15px;font-weight:500;margin:16px 0 6px;color:#444}
.subtitle{color:#666;font-size:14px;margin-bottom:32px}
.badge{display:inline-block;padding:3px 10px;border-radius:20px;
       font-size:12px;font-weight:500;margin-right:6px}
.badge-green{background:#EAF3DE;color:#27500A}
.card{background:#fff;border:0.5px solid #d8d8d2;border-radius:12px;
      padding:24px;margin-bottom:20px}
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
.metric{background:#f4f4f0;border-radius:8px;padding:16px}
.metric-label{font-size:12px;color:#666;margin-bottom:4px}
.metric-value{font-size:24px;font-weight:500;color:#1a1a18}
.metric-sub{font-size:11px;color:#999;margin-top:2px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{background:#f4f4f0;padding:8px 12px;text-align:left;font-weight:500;
   border-bottom:1px solid #e0e0da}
td{padding:8px 12px;border-bottom:0.5px solid #ebebea}
tr:last-child td{border-bottom:none}
.winner-row{background:#EAF3DE}
.narrative{background:#f4f4f2;border-left:3px solid #378ADD;
           padding:16px 20px;border-radius:0 8px 8px 0;
           white-space:pre-wrap;font-size:13px;line-height:1.8;color:#333}
.audit-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.audit-item{font-size:13px}
.audit-key{color:#666;font-size:12px}
.audit-val{font-weight:500}
.anomaly-normal{color:#27500A}
.anomaly-suspicious{color:#854F0B}
.anomaly-anomaly{color:#A32D2D;font-weight:500}
.footer{text-align:center;font-size:12px;color:#999;margin-top:48px;
        padding-top:16px;border-top:0.5px solid #e0e0da}
.chart-wrap{margin:16px 0;min-height:420px}
@media(max-width:600px){.audit-grid{grid-template-columns:1fr}}
"""
        return (
            "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
            f"<meta charset='UTF-8'/>\n"
            f"<meta name='viewport' content='width=device-width,initial-scale=1.0'/>\n"
            f"<title>{title}</title>\n"
            f"<style>{css}</style>\n"
            f"{plotly_js_tag}\n"
            f"</head>\n<body>\n<div class='page'>\n"
            f"{body}\n"
            f"<div class='footer'>Generated by baxter-ts v{version}"
            f" &nbsp;|&nbsp; BAX = Behavioural Analysis &amp; eXplanation</div>\n"
            f"</div>\n</body>\n</html>"
        )
