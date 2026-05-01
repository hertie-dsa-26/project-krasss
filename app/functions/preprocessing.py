# 2 preprocessing.py


# This script handles all data preparation before the data is passed to the model.
# It is composed of one standalone function and four classes organized as follows:

# detect_categorical_columns()   Standalone function. Detects which columns are
#                                categorical and returns their indices. This runs
#                                on the DataFrame before converting to numpy.

# SimpleImputer                  Replaces NaN values with column means.

# StandardScaler                 Standardizes features to mean=0 and std=1.

# OneHotEncoder                  Converts categorical columns into binary columns.
#                                Drops first category to avoid multicollinearity.

# Preprocessor                   Coordinates the three classes above through
#                                composition (not inheritance). This means
#                                Preprocessor does not extend the other classes,
#                                it owns instances of them as internal attributes
#                                and delegates work to them in the correct order:
#                                encode, impute,  scale.
#                                This is the only class that should be called
#                                directly from outside this file.

# Preprocessor uses fit_transform() on training data and transform()
# on validation and test data. This separation ensures that no statistics from
# the test set leak into the training process.

import numpy as np
import pandas as pd


# detect_categorical_columns ():

# This function returns the indices of categorical columns in the feature DataFrame.
# In our dataset this will typically only return the index of 'climate_type_short'.
# All other categorical variables (including state) were previously dropped by prepare_data().

def detect_categorical_columns(X_df: pd.DataFrame) -> list:
    """
    Detects categorical column indices from a DataFrame.

    Args:
        X_df (pd.DataFrame): Feature DataFrame before converting to numpy

    Returns:
        cat_col_indices (list): List of integer indices of categorical columns
    """
    cat_cols = X_df.select_dtypes(
        include=["object", "str", "category"]).columns
    return [X_df.columns.get_loc(c) for c in cat_cols]

# SimpleImputer :

# The SimpleImputer replaces NaN values with the column mean.
# Except for SLEEP (52% NaNs), NaNs are below 10% in all other variables.
# It has 3 methods:
#   fit()           — learns the column means from training data only
#   transform()     — replaces NaNs using the learned means (used on val/test)
#   fit_transform() — learns and replaces in one call (used on train)


class SimpleImputer:
    """
    Replaces NaNs with column means computed from training data only.
    Uses np.where + np.take for efficiency instead of looping over columns.
    """

    def fit(self, X: np.ndarray):
        self.means = np.nanmean(X, axis=0)
        # fallback for all-NaN columns — not our case
        self.means[np.isnan(self.means)] = 0

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        inds = np.where(np.isnan(X))
        X[inds] = np.take(self.means, inds[1])
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

# StandardScaler:

# The StandardScaler follows the same fit/transform structure as SimpleImputer.
# For each feature it subtracts the mean and divides by the standard deviation,
# producing a distribution with mean=0 and std=1.
# This is important for KRR because the kernel is sensitive to feature scale.


class StandardScaler:
    """
    Standardizes features: z = (x - mean) / std
    """

    def fit(self, X: np.ndarray):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # avoid division by zero for constant columns
        self.std[self.std == 0] = 1

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

# The OneHotEncoder:

# The OneHotEncoder uses the indices returned by detect_categorical_columns().
# fit()      — learns the unique categories from training data
# transform() — builds a numerical array from non-categorical columns,
#               then creates a binary block for each category and merges both
# fit_transform() — does both in one call

# We drop the first category per column to avoid multicollinearity.
# Following the fit/transform separation prevents data leakage. The test set is encoded
# using only the categories seen during training, not its own.


class OneHotEncoder:
    """
    Encodes categorical columns as one-hot arrays.
    Drops first category per column to avoid multicollinearity.
    """

    def fit(self, X: np.ndarray, cat_col_indices: list):
        self.cat_col_indices = cat_col_indices
        self.categories = {}
        for idx in cat_col_indices:
            self.categories[idx] = np.unique(X[:, idx][~pd.isnull(X[:, idx])])

    def transform(self, X: np.ndarray) -> np.ndarray:
        num_cols = [i for i in range(
            X.shape[1]) if i not in self.cat_col_indices]
        X_num = X[:, num_cols].astype(float)

        blocks = []
        for idx in self.cat_col_indices:
            cats = self.categories[idx][1:]  # drop first category
            block = np.stack([(X[:, idx] == c).astype(float)
                             for c in cats], axis=1)
            blocks.append(block)

        return np.hstack([X_num] + blocks) if blocks else X_num

    def fit_transform(self, X: np.ndarray, cat_col_indices: list) -> np.ndarray:
        self.fit(X, cat_col_indices)
        return self.transform(X)

# Preprocessor:

# The Preprocessor coordinates the three components above using composition
# it owns instances of OneHotEncoder, SimpleImputer and StandardScaler
# and calls them in the correct order: encode, impute, scale.

# fit_transform() is used on the training set — it learns and applies all transformations.
# transform()     is used on val/test sets — it applies using statistics from training only.

# A "fresh" Preprocessor is created inside each CV fold to prevent statistics
# from one fold leaking into another.


class Preprocessor:
    """
    Coordinates encoding, imputation, scaling.
    Fit on training data, transform on val/test.
    """

    def __init__(self, cat_col_indices: list = None):
        self.cat_col_indices = cat_col_indices or []
        self.encoder = OneHotEncoder()
        self.imputer = SimpleImputer()
        self.scaler = StandardScaler()

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        if self.cat_col_indices:
            X = self.encoder.fit_transform(X, self.cat_col_indices)
        X = self.imputer.fit_transform(X)
        X = self.scaler.fit_transform(X)
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.cat_col_indices:
            X = self.encoder.transform(X)
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        return X
