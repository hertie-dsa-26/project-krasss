import sys
from pathlib import Path

import numpy as np

APP_FUNCTIONS = Path(__file__).resolve().parents[3] / "app" / "functions"
sys.path.insert(0, str(APP_FUNCTIONS))

from assessment import Assessment



def test_r2_score_perfect_preds():
    assessment = Assessment()

    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])

    result = assessment.r2_score(y_true, y_pred)

    assert result == 1.0


def test_r2_score_does_not_exceed_one():
    assessment = Assessment()

    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])

    result = assessment.r2_score(y_true, y_pred)

    assert result <= 1.0


def test_r2_score_mean_baseline_is_zero():
    assessment = Assessment()

    y_true = np.array([1, 2, 3])
    y_pred = np.array([2, 2, 2])

    result = assessment.r2_score(y_true, y_pred)

    assert result == 0.0


def test_r2_score_constant_target_perfect_pred_returns_one():
    assessment = Assessment()

    y_true = np.array([5, 5, 5])
    y_pred = np.array([5, 5, 5])

    result = assessment.r2_score(y_true, y_pred)

    assert result == 1.0


def test_r2_score_constant_target_imperfect_pred_returns_zero():
    assessment = Assessment()

    y_true = np.array([5, 5, 5])
    y_pred = np.array([4, 5, 6])

    result = assessment.r2_score(y_true, y_pred)

    assert result == 0.0


def test_r2_score_can_be_negative():
    assessment = Assessment()

    y_true = np.array([1, 2, 3])
    y_pred = np.array([3, 2, 1])

    result = assessment.r2_score(y_true, y_pred)

    assert result < 0