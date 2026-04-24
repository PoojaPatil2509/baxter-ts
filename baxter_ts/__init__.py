"""
baxter-ts: AutoML time series library with BAX behavioural explanation.

Usage:
    from baxter_ts import BAXModel

    model = BAXModel()
    model.fit(df, target_col="sales", date_col="date")
    model.predict(steps=30)
    model.explain()
    model.anomalies()
    model.visualize()
    model.report("my_report")
"""

from baxter_ts.core import BAXModel

__version__ = "0.1.2"
__all__ = ["BAXModel"]
