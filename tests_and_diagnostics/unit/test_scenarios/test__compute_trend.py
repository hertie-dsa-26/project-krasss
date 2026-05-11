import sys
from pathlib import Path

import pandas as pd
import pytest

APP_FUNCTIONS = Path(__file__).resolve().parents[3] / "app" / "functions"
sys.path.insert(0, str(APP_FUNCTIONS))

from scenarios import _compute_trend



def test_compute_trend_returns_zero_when_less_than_three_rows():
    df = pd.DataFrame({
        "year": [2022, 2023],
        "StateAbbr": ["CA", "CA"],
        "County name": ["A COUNTY", "A COUNTY"],
        "TAVG": [10.0, 12.0],
    })

    result = _compute_trend(df, "A COUNTY", "CA", "TAVG")

    assert result == 0.0


def test_compute_trend_forces_positive_slope_when_appropriate():
    df = pd.DataFrame({
        "year": [2021, 2022, 2023],
        "StateAbbr": ["CA", "CA", "CA"],
        "County name": ["A COUNTY", "A COUNTY", "A COUNTY"],
        "TAVG": [12.0, 11.0, 10.0],
    })

    result = _compute_trend(df, "A COUNTY", "CA", "TAVG")

    assert result == pytest.approx(1.0) # "approx" handles floating point imprecision.


def test_compute_trend_forces_negative_slope_when_appropriate():
    df = pd.DataFrame({
        "year": [2021, 2022, 2023],
        "StateAbbr": ["CA", "CA", "CA"],
        "County name": ["A COUNTY", "A COUNTY", "A COUNTY"],
        "HTDD": [40.0, 45.0, 50.0],
    })

    result = _compute_trend(df, "A COUNTY", "CA", "HTDD")

    assert result == pytest.approx(-5.0) # "approx" handles floating point imprecision.