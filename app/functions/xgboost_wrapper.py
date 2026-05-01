from xgboost import XGBRegressor
import numpy as np

class XGBoostWrapper:
    """
    Wraps XGBRegressor to match the RFFRidgeRegression interface.
    Uses quantile regression for prediction intervals instead of conformal.
    """

    def __init__(self, n_estimators: int = 300, max_depth: int = 4,
                 learning_rate: float = 0.05, seed: int = 42):
        self.seed          = seed
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.learning_rate = learning_rate
        self.cal_residuals_ = None

        # Three models: median, lower quantile, upper quantile
        self._model_median = XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, objective="reg:squarederror",
            random_state=seed
        )
        self._model_lower  = XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, objective="reg:quantileerror",
            quantile_alpha=0.025, random_state=seed
        )
        self._model_upper  = XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, objective="reg:quantileerror",
            quantile_alpha=0.975, random_state=seed
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self._model_median.fit(X_train, y_train)
        self._model_lower.fit(X_train, y_train)
        self._model_upper.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self._model_median.predict(X_test)

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """
        Stores calibration residuals for conformal correction of quantile bounds.
        """
        y_hat               = self.predict(X_cal)
        self.cal_residuals_ = np.abs(y_cal - y_hat)

    def predict_interval(self, X_test: np.ndarray, confidence: float = 0.95) -> tuple:
        """
        Returns (y_pred, lower, upper) using XGBoost quantile regression.
        Optionally corrected with conformal calibration if calibrate() was called.
        """
        y_pred = self._model_median.predict(X_test)
        lower  = self._model_lower.predict(X_test)
        upper  = self._model_upper.predict(X_test)

        # Optional conformal correction
        if self.cal_residuals_ is not None:
            alpha   = 1 - confidence
            n_cal   = len(self.cal_residuals_)
            q_level = min(np.ceil((1 - alpha) * (n_cal + 1)) / n_cal, 1.0)
            q       = np.quantile(self.cal_residuals_, q_level)
            lower   = np.minimum(lower, y_pred - q)
            upper   = np.maximum(upper, y_pred + q)

        return y_pred, lower, upper