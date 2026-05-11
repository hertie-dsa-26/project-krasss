import sys
from pathlib import Path

import numpy as np

APP_FUNCTIONS = Path(__file__).resolve().parents[3] / "app" / "functions"
sys.path.insert(0, str(APP_FUNCTIONS))

from krr import gaussian_kernel, KernelRidgeRegression



def test_gaussian_kernel_returns_expected_shape():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])
    Y = np.array([
        [1.0, 2.0],
        [5.0, 6.0],
        [7.0, 8.0],
    ])

    result = gaussian_kernel(X, Y, sigma2=1.0)

    assert result.shape == (2, 3)


def test_gaussian_kernel_identical_points_have_similarity_one():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    result = gaussian_kernel(X, X, sigma2=1.0)

    np.testing.assert_allclose(np.diag(result), np.array([1.0, 1.0]))


def test_gaussian_kernel_values_are_between_zero_and_one():
    X = np.array([
        [1.0],
        [2.0],
    ])

    result = gaussian_kernel(X, X, sigma2=1.0)

    assert np.all(result >= 0)
    assert np.all(result <= 1)


def test_krr_fit_stores_coefficients_and_train_data():
    model = KernelRidgeRegression(lamb=1e-3, sigma2=1.0)

    X_train = np.array([
        [1.0],
        [2.0],
        [3.0],
    ])
    y_train = np.array([2.0, 4.0, 6.0])

    model.fit(X_train, y_train)

    assert model.coef_ is not None
    assert model.X_train_ is not None
    assert model.coef_.shape == (3,)


def test_krr_predict_returns_one_pred_per_test_row():
    model = KernelRidgeRegression(lamb=1e-3, sigma2=1.0)

    X_train = np.array([
        [1.0],
        [2.0],
        [3.0],
    ])
    y_train = np.array([2.0, 4.0, 6.0])

    X_test = np.array([
        [1.5],
        [2.5],
    ])

    model.fit(X_train, y_train)
    result = model.predict(X_test)

    assert result.shape == (2,)


def test_krr_pred_intervals_return_matching_shapes_and_ordered_bounds():
    model = KernelRidgeRegression(lamb=1e-3, sigma2=1.0)

    X_train = np.array([
        [1.0],
        [2.0],
        [3.0],
    ])
    y_train = np.array([2.0, 4.0, 6.0])

    X_test = np.array([
        [1.5],
        [2.5],
    ])

    model.fit(X_train, y_train)
    y_pred, lower, upper = model.prediction_intervals(X_test)

    assert y_pred.shape == (2,)
    assert lower.shape == (2,)
    assert upper.shape == (2,)
    assert np.all(lower <= y_pred)
    assert np.all(y_pred <= upper)