import sys
from pathlib import Path

import numpy as np

APP_FUNCTIONS = Path(__file__).resolve().parents[3] / "app" / "functions"
sys.path.insert(0, str(APP_FUNCTIONS))

from preprocessing import StandardScaler



def test_standard_scaler_transforms_via_mean_zero_and_std_one():
    scaler = StandardScaler()

    X = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
    ])

    result = scaler.fit_transform(X)

    np.testing.assert_allclose(result.mean(axis=0), np.array([0.0, 0.0]))
    np.testing.assert_allclose(result.std(axis=0), np.array([1.0, 1.0]))


def test_standard_scaler_doesnt_divide_by_zero():
    scaler = StandardScaler()

    X = np.array([
        [5.0, 1.0],
        [5.0, 2.0],
        [5.0, 3.0],
    ])

    result = scaler.fit_transform(X)

    np.testing.assert_allclose(result[:, 0], np.array([0.0, 0.0, 0.0]))
    assert np.all(np.isfinite(result))


def test_transform_uses_train_stats():
    scaler = StandardScaler()

    X_train = np.array([
        [1.0],
        [2.0],
        [3.0],
    ])

    X_test = np.array([
        [100.0],
    ])

    scaler.fit(X_train)
    result = scaler.transform(X_test)

    expected = (X_test - np.array([2.0])) / np.array([np.std([1.0, 2.0, 3.0])])

    np.testing.assert_allclose(result, expected)
