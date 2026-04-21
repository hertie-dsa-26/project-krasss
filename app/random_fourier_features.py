import numpy as np

def rff_features(
    x: np.ndarray,
    R: int,
    sigma: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    r"""Compute Random Fourier Features approximating the Gaussian kernel.

    Draws *R* random frequencies from :math:`N(0, 1)` and phases from
    :math:`U(0, 2\pi)`, then returns the (n, R) feature matrix

    .. math::

        Z = \sqrt{2/R}\,\cos(\sigma\,W\,X^\top + B)

    so that :math:`Z Z^\top \approx K` for the Gaussian kernel with
    bandwidth :math:`\sigma`.
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    x = np.asarray(x, dtype=float).ravel()
    W = rng.randn(R)  # (R,)
    B = rng.uniform(0, 2 * np.pi, size=R)  # (R,)
    Z = np.sqrt(2.0 / R) * np.cos(sigma * np.outer(x, W) + B[None, :])
    return Z

class RFFRidgeRegression:

    def __init__(self, sigma: float = 1.0, lamb: float = 1e-5):
        """
        Initializes the model.

        Attributes:
            coef_ (np.ndarray): The learned coefficients for the features
            sigma (float): the bandwidth for the kernel approximation
            lamb (float): regularization term
        """
        self.sigma = sigma
        self.lamb = lamb
        self.coef_ = None

    def fit(self, X_train:np.ndarray, R: int, y_train:np.ndarray, seed: int = 42):

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

        Z_train = rff_features(X_train, R, sigma = self.sigma, seed = seed)
        self.coef_ = np.linalg.solve(Z_train.T @ Z_train + self.lamb * np.eye(R), Z_train.T @ y_train)
        return self.coef_
    
    def predict(self, X_test:np.ndarray, R: int, seed: int = 42):

        """
        Predicts target values for new data using the trained model.

        Args:
            X_test (np.ndarray): Feature data for making predictions. 
            R (int): Number of random Fourier features to run

        Returns:
            y_pred (np.ndarray): Predicted target values.

        """

        Z_test = rff_features(X_test, R, sigma = self.sigma, seed = seed)
        return Z_test @ self.coef_
        