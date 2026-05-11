import sys
from pathlib import Path

import numpy as np
import pytest

APP_FUNCTIONS = Path(__file__).resolve().parents[3] / "app" / "functions"
sys.path.insert(0, str(APP_FUNCTIONS))

from random_fourier_features import rff_features, RFFRidgeRegression



def test_rff_features_returns_expected_shape():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    result = rff_features(X, R=5, sigma=1.0, seed=42)

    assert result.shape == (3, 5)


def test_rff_features_is_reproducible_with_same_seed():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    result_1 = rff_features(X, R=4, sigma=1.0, seed=42)
    result_2 = rff_features(X, R=4, sigma=1.0, seed=42)

    np.testing.assert_array_equal(result_1, result_2)


def test_rff_model_fit_creates_coefficients():
    model = RFFRidgeRegression(R=5, sigma=1.0, lamb=1e-3)

    X_train = np.array([
        [1.0],
        [2.0],
        [3.0],
    ])
    y_train = np.array([2.0, 4.0, 6.0])

    model.fit(X_train, y_train)

    assert model.coef_ is not None
    assert model.coef_.shape == (5,)


def test_rff_model_predict_returns_one_pred_per_row():
    model = RFFRidgeRegression(R=5, sigma=1.0, lamb=1e-3)

    X_train = np.array([
        [1.0],
        [2.0],
        [3.0],
    ])
    y_train = np.array([2.0, 4.0, 6.0])
    X_test = np.array([
        [4.0],
        [5.0],
    ])

    model.fit(X_train, y_train)
    result = model.predict(X_test)

    assert result.shape == (2,)


def test_rff_pred_interval_raises_before_calibration():
    model = RFFRidgeRegression(R=5, sigma=1.0, lamb=1e-3)

    X_train = np.array([
        [1.0],
        [2.0],
        [3.0],
    ])
    y_train = np.array([2.0, 4.0, 6.0])
    X_test = np.array([
        [4.0],
    ])

    model.fit(X_train, y_train)

    with pytest.raises(RuntimeError, match="calibrate"):
        model.predict_interval(X_test)


def test_rff_predict_interval_bounds_contain_preds():
    model = RFFRidgeRegression(R=5, sigma=1.0, lamb=1e-3)

    X_fit = np.array([
        [1.0],
        [2.0],
        [3.0],
    ])
    y_fit = np.array([2.0, 4.0, 6.0])

    X_cal = np.array([
        [1.5],
        [2.5],
    ])
    y_cal = np.array([3.0, 5.0])

    X_test = np.array([
        [4.0],
        [5.0],
    ])

    model.fit(X_fit, y_fit)
    model.calibrate(X_cal, y_cal)

    y_pred, lower, upper = model.predict_interval(X_test)

    assert y_pred.shape == (2,)
    assert lower.shape == (2,)
    assert upper.shape == (2,)
    assert np.all(lower <= y_pred)
    assert np.all(y_pred <= upper)