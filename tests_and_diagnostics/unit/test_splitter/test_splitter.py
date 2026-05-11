import sys
from pathlib import Path

import numpy as np

APP_FUNCTIONS = Path(__file__).resolve().parents[3] / "app" / "functions"
sys.path.insert(0, str(APP_FUNCTIONS))

from splitter import Splitter



def test_splitter_sets_train_and_test_years():
    X = np.array([
        [10.0],
        [20.0],
        [30.0],
        [40.0],
    ])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    years = np.array([2020, 2021, 2022, 2023])

    splitter = Splitter(X, y, years)

    np.testing.assert_array_equal(splitter.train_years, np.array([2020, 2021, 2022]))
    assert splitter.test_year == 2023


def test_time_series_splits_expand_train_years_incrementally():
    X = np.array([
        [10.0],
        [20.0],
        [30.0],
        [40.0],
    ])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    years = np.array([2020, 2021, 2022, 2023])

    splitter = Splitter(X, y, years)
    splits = splitter.time_series_splits()

    assert len(splits) == 2

    X_train_0, X_val_0, y_train_0, y_val_0 = splits[0]
    np.testing.assert_array_equal(X_train_0, np.array([[10.0]]))
    np.testing.assert_array_equal(X_val_0, np.array([[20.0]]))
    np.testing.assert_array_equal(y_train_0, np.array([1.0]))
    np.testing.assert_array_equal(y_val_0, np.array([2.0]))

    X_train_1, X_val_1, y_train_1, y_val_1 = splits[1]
    np.testing.assert_array_equal(X_train_1, np.array([[10.0], [20.0]]))
    np.testing.assert_array_equal(X_val_1, np.array([[30.0]]))
    np.testing.assert_array_equal(y_train_1, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(y_val_1, np.array([3.0]))


def test_get_test_split_uses_last_year_as_test():
    X = np.array([
        [10.0],
        [20.0],
        [30.0],
        [40.0],
    ])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    years = np.array([2020, 2021, 2022, 2023])

    splitter = Splitter(X, y, years)

    X_train, X_test, y_train, y_test = splitter.get_test_split()

    np.testing.assert_array_equal(X_train, np.array([[10.0], [20.0], [30.0]]))
    np.testing.assert_array_equal(X_test, np.array([[40.0]]))
    np.testing.assert_array_equal(y_train, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(y_test, np.array([4.0]))