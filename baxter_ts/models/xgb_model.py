"""XGBoost model wrapper."""

from xgboost import XGBRegressor
from baxter_ts.models.base_model import BaseTimeSeriesModel


class XGBModel(BaseTimeSeriesModel):
    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05,
                 max_depth: int = 6, random_state: int = 42):
        super().__init__("XGBoost")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def _build_model(self):
        return XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbosity=0,
            n_jobs=-1,
        )
