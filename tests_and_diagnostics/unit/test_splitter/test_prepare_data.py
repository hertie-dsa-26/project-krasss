import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP_FUNCTIONS = Path(__file__).resolve().parents[3] / "app" / "functions"
sys.path.insert(0, str(APP_FUNCTIONS))

from splitter import prepare_data



def test_prepare_data_keeps_target_and_drops_intended_vars():
    df = pd.DataFrame({
        "year": [2020, 2021],
        "StateAbbr": ["CA", "CA"],
        "County name": ["A", "A"],
        "CountyFIPS": [1, 1],
        "STATION": ["S1", "S1"],
        "STATION_NAME": ["Station", "Station"],
        "BPHIGH": [1.0, 2.0],
        "CASTHMA": [3.0, 4.0],
        "COPD": [5.0, 6.0],
        "MHLTH": [7.0, 8.0],
        "PHLTH": [9.0, 10.0],
        "SLEEP": [11.0, 12.0],
        "STROKE": [13.0, 14.0],
        "TAVG": [70.0, 71.0],
        "climate_type_short": ["dry", "wet"],
    })

    X, y, years = prepare_data(df, target="MHLTH")

    assert "MHLTH" not in X.columns
    assert "TAVG" in X.columns
    assert "climate_type_short" in X.columns

    dropped_columns = {
        "StateAbbr", "County name", "CountyFIPS", "STATION", "STATION_NAME",
        "BPHIGH", "CASTHMA", "COPD", "PHLTH", "SLEEP", "STROKE", "year",
    }

    assert dropped_columns.isdisjoint(X.columns)
    np.testing.assert_array_equal(y, np.array([7.0, 8.0]))
    np.testing.assert_array_equal(years, np.array([2020, 2021]))


def test_prepare_data_drops_rows_missing_target():
    df = pd.DataFrame({
        "year": [2020, 2021, 2022],
        "StateAbbr": ["CA", "CA", "CA"],
        "County name": ["A", "A", "A"],
        "CountyFIPS": [1, 1, 1],
        "STATION": ["S1", "S1", "S1"],
        "STATION_NAME": ["Station", "Station", "Station"],
        "BPHIGH": [1.0, 2.0, 3.0],
        "CASTHMA": [3.0, 4.0, 5.0],
        "COPD": [5.0, 6.0, 7.0],
        "MHLTH": [7.0, np.nan, 9.0],
        "PHLTH": [9.0, 10.0, 11.0],
        "SLEEP": [11.0, 12.0, 13.0],
        "STROKE": [13.0, 14.0, 15.0],
        "TAVG": [70.0, 71.0, 72.0],
    })

    X, y, years = prepare_data(df, target="MHLTH")

    assert len(X) == 2
    np.testing.assert_array_equal(y, np.array([7.0, 9.0]))
    np.testing.assert_array_equal(years, np.array([2020, 2022]))