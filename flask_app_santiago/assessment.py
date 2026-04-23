# assessment.py

# This script is in charge of evaluating model performance.
# It contains one class with two methods:

# r2_score()           Measures how much variance the model explains.
#                      1 = perfect, 0 = predicting the mean, < 0 = worse than the mean.

# mean_squared_error() Measures the average squared difference between
#                      predicted and actual values. Used for hyperparameter tuning.

import numpy as np


class Assessment:

    def r2_score(self, y_test: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the R² score.
        (1) Calculate the mean of true values
        (2) Calculate the total sum of squares (TSS)
        (3) Calculate the residual sum of squares (RSS)
        (4) 1 - (RSS / TSS)

        Args:
            y_test (np.ndarray): True values
            y_pred (np.ndarray): Predicted values

        Returns:
            score (float): R² score — a number <= 1, with 1 being perfect
        """
        mean_y = np.mean(y_test)
        TSS = np.sum((y_test - mean_y) ** 2)
        RSS = np.sum((y_test - y_pred) ** 2)
        return 1 - (RSS / TSS)

    def mean_squared_error(self, y_test: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Mean Squared Error (MSE).
        (1) Find the difference between true and predicted values
        (2) Square each difference to eliminate negatives and penalize large errors
        (3) Compute the mean of the squared differences

        Args:
            y_test (np.ndarray): True values
            y_pred (np.ndarray): Predicted values

        Returns:
            score (float): Average squared difference between actual and predicted values
        """
        return np.square(np.subtract(y_test, y_pred)).mean()
