"""CatBoost model wrapper."""

from catboost import CatBoostRegressor
from baxter_ts.models.base_model import BaseTimeSeriesModel


class CatModel(BaseTimeSeriesModel):
    def __init__(self, iterations: int = 300, learning_rate: float = 0.05,
                 depth: int = 6, random_state: int = 42):
        super().__init__("CatBoost")
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.random_state = random_state

    def _build_model(self):
        return CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            random_seed=self.random_state,
            verbose=0,
        )
