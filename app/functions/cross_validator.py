# 5 cross_validator.py

# This script is in charge of running cross validation across time series folds.
# It coordinates the Splitter, Preprocessor, KRR model and Assessment together.

# A fresh Preprocessor is created inside each fold

# start_fold gives us the option to skip early unstable folds where the training set is too small.
# In our data the jump from 255 to 896 rows per year at 2018 makes
# early folds unreliable for hyperparameter selection.

# cross_val_score() returns a dictionary with keys 'mse' and 'r2':

#   scores["mse"] — used for hyperparameter tuning in the grid search
#   scores["r2"]  — used for reporting and interpretation

import numpy as np
from preprocessing import Preprocessor
from assessment import Assessment


class CrossValidator:

    def __init__(self, start_fold: int = 0):
        """
        Args:
            start_fold (int): Only score folds >= this index.
                              By default set to 0 (all folds scored).
        """
        self.start_fold = start_fold
        self.assessment = Assessment()

    def cross_val_score(self, model, splitter, preprocessor: Preprocessor) -> dict:
        """
        Evaluates the model across time series folds using MSE and R².

        Args:
            model:        KRR model — must have fit() and predict() methods
            splitter:     Splitter instance — provides time_series_splits()
            preprocessor: Preprocessor instance — provides cat_col_indices

        Returns:
            dict with keys 'mse' and 'r2' — each a list of scores per scored fold
        """
        # I'll explain the folowing loop because it can be tricky at first
        # The time_series_splits() function from the Splitter class returns a list of tuples:

        # (X_train_fold_0, X_val_fold_0, y_train_fold_0, y_val_fold_0),
        # (...)
        # (X_train_fold_7, X_val_fold_7, y_train_fold_7, y_val_fold_7)

        # Each iteration unpacks one tuple into 4 arrays corresponding to one fold.
        # For each fold we fit_transform on training data and transform only on validation.
        # We are, in other words, iterating across each fold

        mse_scores = []
        r2_scores = []

        for i, (X_train_f, X_val_f, y_train_f, y_val_f) in enumerate(splitter.time_series_splits()):

            # A fresh Preprocessor is created per fold to prevent statistics
            # from one fold leaking into the next
            preprocessor_fold = Preprocessor(
                cat_col_indices=preprocessor.cat_col_indices)
            X_tr = preprocessor_fold.fit_transform(X_train_f)
            X_v = preprocessor_fold.transform(X_val_f)

            model.fit(X_tr, y_train_f)
            y_pred = model.predict(X_v)

            if i >= self.start_fold:
                mse_scores.append(
                    self.assessment.mean_squared_error(y_val_f, y_pred))
                r2_scores.append(self.assessment.r2_score(y_val_f, y_pred))

        return {"mse": mse_scores, "r2": r2_scores}
