import sys
from pathlib import Path

import numpy as np

APP_FUNCTIONS = Path(__file__).resolve().parents[3] / "app" / "functions"
sys.path.insert(0, str(APP_FUNCTIONS))

from assessment import Assessment



def test_mean_squared_error_expected_value():
    assessment = Assessment()

    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 5])

    result = assessment.mean_squared_error(y_true, y_pred)

    assert result == 4 / 3


def test_mean_squared_error_perfect_predictions_is_zero():
    assessment = Assessment()

    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])

    result = assessment.mean_squared_error(y_true, y_pred)

    assert result == 0.0


def test_mean_squared_error_handles_negative_values():
    assessment = Assessment()

    y_true = np.array([-1, 0, 1])
    y_pred = np.array([0, 0, 0])

    result = assessment.mean_squared_error(y_true, y_pred)

    assert result == 2 / 3


def test_mean_squared_error_penalizes_large_errors_quadratically():
    assessment = Assessment()

    y_true = np.array([0, 0])
    y_pred = np.array([1, 3])

    result = assessment.mean_squared_error(y_true, y_pred)

    assert result == 5.0