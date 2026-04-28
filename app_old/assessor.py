import numpy as np

class Assessment:
    
    def r2_score(self, y_test:np.ndarray, y_pred:np.ndarray) -> float:
        '''
        Calculate the r2 score. 
        (1) Calculate the mean of true values
        (2) Calculate the total sum of squares (TSS)
        (3) Calculate the residual sum of squares (RSS)
        (4) 1 - (RSS / TSS)

        Args: 
            y_test (np.ndarray): true values for the outcome 
            y_pred (np.ndarray): predicted values for the outcome given our model 

        Returns: 
            accuracy (float): r^2 score for a given model which is a number a <= 1 (with 1 being best)
        '''
        mean_y_test = np.mean(y_test)
        TSS = np.sum((y_test - mean_y_test) ** 2)
        RSS = np.sum((y_test - y_pred) ** 2)
        return 1 - (RSS/TSS)

    def mean_squared_error(self, y_test:np.ndarray, y_pred:np.ndarray) -> float:
        '''
        Calculate the Mean Squared Error (MSE) by 
        (1) finding the difference between the true values (y_test) and the predicted values (y_pred). 
        (2) square each of these differences to eliminate negative values and emphasize larger errors 
        (3) compute the mean of these squared differences to obtain the MSE.

        Args: 
            y_test (np.ndarray): true values for the outcome 
            y_pred (np.ndarray): predicted values for the outcome given our model 

        Returns: 
            MSE (float): the average squared difference between the actual and predicted values 
        '''
        
        return np.square(np.subtract(y_test, y_pred)).mean()