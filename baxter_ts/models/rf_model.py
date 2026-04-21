"""Random Forest model wrapper."""

from sklearn.ensemble import RandomForestRegressor
from baxter_ts.models.base_model import BaseTimeSeriesModel


class RFModel(BaseTimeSeriesModel):
    def __init__(self, n_estimators: int = 200, max_depth: int = 10,
                 random_state: int = 42):
        super().__init__("RandomForest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def _build_model(self):
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            n_jobs=-1,
            random_state=self.random_state,
        )
