import sys
from pathlib import Path

import numpy as np

APP_FUNCTIONS = Path(__file__).resolve().parents[3] / "app" / "functions"
sys.path.insert(0, str(APP_FUNCTIONS))

from cross_validator import CrossValidator
from preprocessing import Preprocessor
from splitter import Splitter



class MeanModel:
    def fit(self, X_train, y_train):
        self.mean_ = np.mean(y_train)

    def predict(self, X_val):
        return np.full(X_val.shape[0], self.mean_)
# We define this class because CrossValidator() expects a model object with two methods:
# .fit() and .predict(). 
# Does CrossValidator correctly loop through folds, preprocess data, fit a model, predict, 
# and return MSE/R^2 scores?


def test_cross_validator_returns_mse_and_r2_for_each_fold():
    X = np.array([
        [1.0],
        [1.5],
        [2.0],
        [2.5],
        [3.0],
        [3.5],
        [4.0],
        [4.5],
    ])

    y = np.array([
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
    ])

    years = np.array([
        2020, 2020,
        2021, 2021,
        2022, 2022,
        2023, 2023,
    ])

    splitter = Splitter(X, y, years)
    preprocessor = Preprocessor()
    model = MeanModel()
    cv = CrossValidator(start_fold=0)

    result = cv.cross_val_score(model, splitter, preprocessor)

    assert set(result.keys()) == {"mse", "r2"}
    assert len(result["mse"]) == 2
    assert len(result["r2"]) == 2
    assert all(np.isfinite(score) for score in result["mse"])
    assert all(np.isfinite(score) for score in result["r2"])