import numpy as np
from xgboost import XGBRegressor

def rff_features(x: np.ndarray, R: int, sigma: float = 1.0, seed: int = 42,) -> np.ndarray:
    """Compute Random Fourier Features approximating the Gaussian kernel.

    Draws *R* random frequencies from N(0, 1) and phases from
    U(0, 2π), then returns the (n, R) feature matrix

    .. math::

        Z = sqrt(2/R) * cos((sigma * W * X^T) + B)

    so that Z * Z^T is approximately K for the Gaussian kernel with
    bandwidth sigma.
    """
    rng = np.random.RandomState(seed)
    x = np.atleast_2d(np.asarray(x, dtype=float)) 
    W = rng.randn(x.shape[1], R)                    # (d, R)
    B = rng.uniform(0, 2 * np.pi, size=R)           # (R,)
    Z = np.sqrt(2.0 / R) * np.cos(x @ W * sigma + B[None, :])
    return Z

class RFFRidgeRegression:

    def __init__(self, sigma: float = 1.0, lamb: float = 1e-5, R: int = 100, seed: int = 42):
        """
        Initializes the model.

        Attributes:
            coef_ (np.ndarray): The learned coefficients for the features
            sigma (float): the bandwidth for the kernel approximation
            lamb (float): regularization term
            R (int): number of random Fourier features
            seed (int): random seed for reproducibility
            cal_residuals_ (np.ndarray): stored calibration residuals
        """
        self.sigma = sigma
        self.lamb  = lamb
        self.R     = R
        self.seed  = seed
        self.coef_ = None
        self.cal_residuals_ = None

    def fit(self, X_train:np.ndarray, y_train:np.ndarray):

        """
        The fit function computes the coefficients by running Random Fourier Features to approximate KRR:

        The function proceeds as follows:

        (1) Computes Z by drawing R random frequencies such that ZZ^T is approximately equal to K
        (2) Runs ordinary ridge regression on Z instead of solving the kernel system
        (3) Stores the coefficients that can be used for predictions


        Args:
            X_train (np.ndarray): Training feature data. 
            y_train (np.ndarray): Training target values.
            R (int): Number of random Fourier features to run

        Returns: 
            Sets and returns the coefficients
        """

        Z_train = rff_features(X_train, self.R, sigma = self.sigma, seed = self.seed)
        self.coef_ = np.linalg.solve(Z_train.T @ Z_train + self.lamb * np.eye(self.R), Z_train.T @ y_train)
    
    def predict(self, X_test:np.ndarray):

        """
        Predicts target values for new data using the trained model.

        Args:
            X_test (np.ndarray): Feature data for making predictions. 
            R (int): Number of random Fourier features to run

        Returns:
            y_pred (np.ndarray): Predicted target values.

        """

        Z_test = rff_features(X_test, self.R, sigma = self.sigma, seed = self.seed)
        return Z_test @ self.coef_
    
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """
        Computes and stores nonconformity scores on a held-out calibration set.
        Must be called after fit() and before predict_interval().

        Args:
            X_cal (np.ndarray): Calibration feature data (m, d)
            y_cal (np.ndarray): Calibration target values (m,)
        """
        if self.coef_ is None:
            raise RuntimeError("Model must be fitted before calibration.")
        y_hat = self.predict(X_cal)
        self.cal_residuals_ = np.abs(y_cal - y_hat)  # nonconformity scores

    def predict_interval(self, X_test: np.ndarray, confidence: float = 0.9) -> tuple:
        """
        Generates conformal prediction intervals for new data.
        Requires calibrate() to have been called first.

        The interval is: [ŷ - q, ŷ + q]
        where q is the (1-α) quantile of calibration residuals.

        Args:
            X_test     (np.ndarray): Feature data (m, d)
            confidence (float): e.g. 0.95 for 95% coverage guarantee

        Returns:
            (y_pred, lower, upper): each np.ndarray of shape (m,)
        """
        if self.cal_residuals_ is None:
            raise RuntimeError("Must call calibrate() before predict_interval().")

        alpha  = 1 - confidence
        n_calib  = len(self.cal_residuals_)

        # Finite-sample corrected quantile (Angelopoulos & Bates, 2022)
        q_level = np.ceil((1 - alpha) * (n_calib + 1)) / n_calib
        q_level = min(q_level, 1.0)
        q       = np.quantile(self.cal_residuals_, q_level)

        y_pred  = self.predict(X_test)
        return y_pred, y_pred - q, y_pred + q