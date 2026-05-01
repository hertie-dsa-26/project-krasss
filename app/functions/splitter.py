# 1 splitter.py


import numpy as np
import pandas as pd

# Global variables and non features. Each model requires only one health var
health_vars = ['BPHIGH', 'CASTHMA', 'COPD',
               'MHLTH', 'PHLTH', 'SLEEP', 'STROKE']
drop_non_features = ["County name", "CountyFIPS",
                     "STATION", "STATION_NAME", "StateAbbr"]

# Adding also data types of the input


def prepare_data(df: pd.DataFrame, target: str):
    """
    Prepares features X, target y, and year labels from the full dataframe.

    Args:
        df (pd.DataFrame): Full dataset
        target (str):      Target health variable e.g. 'COPD'

    Returns:
        X (pd.DataFrame):     Feature matrix (without target, year, identifiers)
        y (np.ndarray):       Target values
        years (np.ndarray):   Year per row
    """
    df = df.copy()
    drop_health = [col for col in health_vars if col != target]
    df = df.drop(columns=drop_health + drop_non_features)

    # Drop rows where target is NaN. This prevents us from imputing the y variable.
    # In the case of SLEEP this would shrink the dataset only to rows with actual sleep data.
    df = df.dropna(subset=[target])

    X = df.drop(columns=[target, "year"])
    y = df[target].values
    years = df["year"].values  # We need these values to create the folds
    return X, y, years


class Splitter:
    # This class was initially called "Model" but for clarity I changed the name to Splitter.
    # It returns the folds needed for cross validation as well as the full X and y.
    # Cross validation was initially part of this class but was moved to Assessment
    # to prevent confusion and keep responsibilities separate.

    def __init__(self, X: np.ndarray, y: np.ndarray, years: np.ndarray):
        """
        Args:
            X (np.ndarray):     Feature matrix
            y (np.ndarray):     Target values
            years (np.ndarray): Year label per row. Used to create time-based splits.
        """
        self.X = X
        self.y = y
        self.years = years

        # [2013, 2014, ..., 2023]
        self.unique_years = np.sort(np.unique(years))
        self.train_years = self.unique_years[:-1]      # 2013-2022
        self.test_year = self.unique_years[-1]        # 2023

    def time_series_splits(self):
        """
        Each fold adds one more year to training and validates on the next unseen year.
        We start in 2014 because there is very little data for 2013.

        e.g. with years [2014..2022]:
            fold 0: train=2014,           validate=2015
            fold 1: train=2014-2015,      validate=2016
            ...
            fold 7: train=2014-2021,      validate=2022

        The 2023 data corresponds to the test set.

        Returns:
            splits (list of tuples): (X_train, X_val, y_train, y_val) per fold.
            Each row is one fold, and in each fold we have our 4 corresponding datasets.
        """
        splits = []
        for i in range(1, len(self.train_years)):  # Iterating from 2015 to 2022
            train_mask = np.isin(self.years, self.train_years[:i])
            val_mask = self.years == self.train_years[i]

            X_train, y_train = self.X[train_mask], self.y[train_mask]
            X_val,   y_val = self.X[val_mask],   self.y[val_mask]

            splits.append((X_train, X_val, y_train, y_val))

        return splits

    def get_test_split(self):
        """
        Returns full 2013-2022 training data and the held-out 2023 test set.

        Returns:
            X_train, X_test, y_train, y_test (np.ndarray)
        """
        train_mask = np.isin(self.years, self.train_years)
        test_mask = self.years == self.test_year

        return (
            self.X[train_mask], self.X[test_mask],
            self.y[train_mask], self.y[test_mask]
        )
