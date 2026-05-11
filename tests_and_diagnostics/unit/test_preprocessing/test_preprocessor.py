import sys
from pathlib import Path

import numpy as np

APP_FUNCTIONS = Path(__file__).resolve().parents[3] / "app" / "functions"
sys.path.insert(0, str(APP_FUNCTIONS))

from preprocessing import Preprocessor



def test_preprocessor_num_data_imputes_and_scales():
    preprocessor = Preprocessor()

    X = np.array([
        [1.0, 10.0],
        [3.0, np.nan],
        [5.0, 30.0],
    ])

    result = preprocessor.fit_transform(X)

    assert result.shape == (3, 2)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result.mean(axis=0), np.array([0.0, 0.0]))
    np.testing.assert_allclose(result.std(axis=0), np.array([1.0, 1.0]))


def test_preprocessor_with_cat_col_returns_num_array():
    preprocessor = Preprocessor(cat_col_indices=[1])

    X = np.array([
        [10.0, "dry"],
        [np.nan, "wet"],
        [30.0, "dry"],
    ], dtype=object)

    result = preprocessor.fit_transform(X)

    assert result.shape == (3, 2)
    assert np.issubdtype(result.dtype, np.number)
    assert np.all(np.isfinite(result))


def test_preprocessor_transform_uses_train_pipeline_on_new_data():
    preprocessor = Preprocessor(cat_col_indices=[1])

    X_train = np.array([
        [10.0, "dry"],
        [20.0, "wet"],
        [30.0, "dry"],
    ], dtype=object)

    X_test = np.array([
        [40.0, "wet"],
    ], dtype=object)

    preprocessor.fit_transform(X_train)
    result = preprocessor.transform(X_test)

    assert result.shape == (1, 2)
    assert np.issubdtype(result.dtype, np.number)
    assert np.all(np.isfinite(result))