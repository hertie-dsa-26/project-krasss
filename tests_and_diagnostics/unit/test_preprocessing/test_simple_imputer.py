import sys
from pathlib import Path

import numpy as np

APP_FUNCTIONS = Path(__file__).resolve().parents[3] / "app" / "functions"
sys.path.insert(0, str(APP_FUNCTIONS))

from preprocessing import SimpleImputer



def test_simple_imputer_with_nan():
    imputer = SimpleImputer()

    X = np.array([
        [1.0, 10.0],
        [3.0, np.nan],
        [5.0, 30.0],
    ])

    result = imputer.fit_transform(X)

    expected = np.array([
        [1.0, 10.0],
        [3.0, 20.0],
        [5.0, 30.0],
    ])

    np.testing.assert_array_equal(result, expected)


def test_simple_imputer_replaces_all_nan_cols_with_zero():
    imputer = SimpleImputer()

    X = np.array([
        [np.nan, 1.0],
        [np.nan, 3.0],
        [np.nan, 5.0],
    ])

    result = imputer.fit_transform(X)

    expected = np.array([
        [0.0, 1.0],
        [0.0, 3.0],
        [0.0, 5.0],
    ])

    np.testing.assert_array_equal(result, expected)
    # The all-NaN imputer edge case passes, but NumPy sends a warning before the fallback in 
    # preprocessing.py replaces the missing column mean with 0.
