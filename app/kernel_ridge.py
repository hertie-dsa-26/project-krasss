import numpy as np
from scipy.stats import norm

def gaussian_kernel(x: np.ndarray, y: np.ndarray, sigma2: float = 2.0) -> np.ndarray:
    """Compute the Gaussian (RBF) kernel matrix between 1-D+ arrays *x* and *y*.

    Returns an (n, m) matrix where entry (i, j) is
    exp(-||x_i - y_j||^2 / (2 * sigma2)).
    """
    x_sq = np.sum(x**2, axis=1, keepdims=True)  # (n, 1)
    y_sq = np.sum(y**2, axis=1, keepdims=True)  # (m, 1)
    sq_dists = x_sq + y_sq.T - 2 * (x @ y.T)        # (n, m)
    # clip floating point negatives to 0
    sq_dists = np.maximum(sq_dists, 0)
    return np.exp(-sq_dists / (2 * sigma2))

class KernelRidgeRegression:
    """
    Implements Kernel Ridge Regression using NumPy.

    This class provides methods to fit a kernel ridge regression model
    to training data and make predictions on new data. 
    """
    def __init__(self, lamb: float = 1e-5, sigma2: float = 0.5):
        """
        Initializes the model.

        Attributes:
            coef_ (np.ndarray): The learned coefficients for the features
            sigma2 (float): parameter for the RBF kernel
            lamb (float): regularisation term
        """
        self.lamb = lamb
        self.sigma2 = sigma2
        self.coef_ = None
        self.X_train_ = None

    def fit(self, X_train:np.ndarray, y_train:np.ndarray):
        """
        The fit function computes the dual coefficients a by solving the linear system:

        (K(X,X)+λI)a=y

        Here, X and y represent the training data, 
                K(X,X) is the Gram matrix computed from compute_gaussian_kernel, 
                λ is the regularization parameter, and 
                I is the identity matrix of size n, n is length of y_train

        The function proceeds as follows:

        (1) Computes the symmetric Gram matrix and stores it in the variable k_train, where k_train is size nxn where n is the length of y_train.
        (2) Constructs the left-hand side matrix a:=K(X,X)+λI
        (3) Solves the resulting linear system for a.
        (4) stores x_train and the computed alpha inside the model. Keeping x_train is essential for future predictions on new inputs.


        Args:
            X_train (np.ndarray): Training feature data. 
            y_train (np.ndarray): Training target values.

        Returns: 
            not so much as setting the dual coef that we will use to predict 
        """
        
        K_train = gaussian_kernel(X_train, X_train, sigma2=self.sigma2)
        A = K_train + self.lamb * np.eye(len(X_train))
        self.coef_ = np.linalg.solve(A, y_train)
        self.X_train_ = X_train
        # store K_inv once during training to speed up use during calculation of prediction intervals
        self.K_inv_ = np.linalg.inv(A) 
        return self.coef_


    def predict(self, X_test:np.ndarray) -> np.ndarray:
        """
        Predicts target values for new data using the trained model.

        Args:
            X_test (np.ndarray): Feature data for making predictions. 

        Returns:
            y_pred (np.ndarray): Predicted target values.

        """
        K_test = gaussian_kernel(X_test, self.X_train_, sigma2=self.sigma2)
        return K_test @ self.coef_
    
    
    def prediction_intervals(self, X_test: np.ndarray,
    confidence: float = 0.95) -> tuple:
        
        """
        Setting up prediction intervals.

        Bootstrapping might be methodologically more robust but hard to do with a stored model
        and would also need a lot more time to re-run every time if we don't store the model.

        So for efficiency reasons, this Bayesian approach using a Gaussian Process might be faster.
        """

        K_star  = gaussian_kernel(X_test, self.X_train_, sigma2=self.sigma2)
        K_ss    = gaussian_kernel(X_test, X_test, sigma2=self.sigma2)
        var     = np.diag(K_ss - K_star @ self.K_inv_ @ K_star.T)
        var     = np.maximum(var, 0)
        std     = np.sqrt(var)
        z       = norm.ppf(1 - (1 - confidence) / 2)
        y_pred  = self.predict(X_test)
        return y_pred, y_pred - z * std, y_pred + z * std