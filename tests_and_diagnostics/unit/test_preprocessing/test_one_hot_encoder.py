import sys
from pathlib import Path

import numpy as np

APP_FUNCTIONS = Path(__file__).resolve().parents[3] / "app" / "functions"
sys.path.insert(0, str(APP_FUNCTIONS))

from preprocessing import OneHotEncoder



def test_one_hot_encoder_drops_first_cat_and_keeps_numeric_cols():
    encoder = OneHotEncoder()

    X = np.array([
        [10.0, "dry"],
        [20.0, "wet"],
        [30.0, "dry"],
    ], dtype=object)

    result = encoder.fit_transform(X, cat_col_indices=[1])

    expected = np.array([
        [10.0, 0.0],
        [20.0, 1.0],
        [30.0, 0.0],
    ])

    np.testing.assert_array_equal(result, expected)


def test_one_hot_encoder_handles_multiple_cats():
    encoder = OneHotEncoder()

    X = np.array([
        [10.0, "dry"],
        [20.0, "wet"],
        [30.0, "humid"],
        [40.0, "dry"],
    ], dtype=object)

    result = encoder.fit_transform(X, cat_col_indices=[1])

    expected = np.array([
        [10.0, 0.0, 0.0],
        [20.0, 0.0, 1.0],
        [30.0, 1.0, 0.0],
        [40.0, 0.0, 0.0],
    ])

    np.testing.assert_array_equal(result, expected)


def test_one_hot_encoder_unseen_cat_becomes_zero():
    encoder = OneHotEncoder()

    X_train = np.array([
        [10.0, "dry"],
        [20.0, "wet"],
    ], dtype=object)

    X_test = np.array([
        [30.0, "snow"],
    ], dtype=object)

    encoder.fit(X_train, cat_col_indices=[1])
    result = encoder.transform(X_test)

    expected = np.array([
        [30.0, 0.0],
    ])

    np.testing.assert_array_equal(result, expected)