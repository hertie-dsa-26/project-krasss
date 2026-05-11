import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

APP_FUNCTIONS = Path(__file__).resolve().parents[3] / "app" / "functions"
sys.path.insert(0, str(APP_FUNCTIONS))

from scenarios import generate_scenario



def make_scenario_df():
    return pd.DataFrame({
        "year": [2021, 2022, 2023],
        "StateAbbr": ["CA", "CA", "CA"],
        "County name": ["A COUNTY", "A COUNTY", "A COUNTY"],
        "CountyFIPS": [1, 1, 1],
        "STATION": ["S1", "S1", "S1"],
        "STATION_NAME": ["Station", "Station", "Station"],
        "BPHIGH": [1.0, 1.0, 1.0],
        "CASTHMA": [2.0, 2.0, 2.0],
        "COPD": [3.0, 3.0, 3.0],
        "MHLTH": [4.0, 4.0, 4.0],
        "PHLTH": [5.0, 5.0, 5.0],
        "SLEEP": [6.0, 6.0, 6.0],
        "STROKE": [7.0, 7.0, 7.0],
        "TAVG": [10.0, 11.0, 12.0],
        "TMAX": [20.0, 21.0, 22.0],
        "TMIN": [5.0, 6.0, 7.0],
        "CLDD": [100.0, 110.0, 120.0],
        "HTDD": [50.0, 45.0, 40.0],
        "DT100": [1.0, 2.0, 3.0],
        "DX90": [4.0, 5.0, 6.0],
        "EMXT": [30.0, 31.0, 32.0],
        "EMNT": [0.0, -1.0, -2.0],
        "PRCP": [1.0, 1.0, 1.0],
        "total_population": [1000, 1000, 1000],
        "climate_type_short": ["dry", "dry", "dry"],
    })


def test_generate_scenario_returns_expected_future_years():
    df = make_scenario_df()

    X_future, future_years = generate_scenario(
        df,
        county_name="A COUNTY",
        state_abbr="CA",
        scenario_key="middle_road",
        horizon=3,
        baseline_yr=2023,
    )

    assert future_years == [2024, 2025, 2026]
    assert len(X_future) == 3


def test_generate_scenario_drops_non_feature_columns():
    df = make_scenario_df()

    X_future, _ = generate_scenario(
        df,
        county_name="A COUNTY",
        state_abbr="CA",
        scenario_key="middle_road",
        horizon=1,
        baseline_yr=2023,
    )

    dropped_columns = {
        "year", "StateAbbr", "County name", "CountyFIPS",
        "STATION", "STATION_NAME",
        "BPHIGH", "CASTHMA", "COPD", "MHLTH", "PHLTH", "SLEEP", "STROKE",
    }

    assert dropped_columns.isdisjoint(X_future.columns)
    assert "TAVG" in X_future.columns
    assert "climate_type_short" in X_future.columns


def test_generate_scenario_projects_warming_vars_forward():
    df = make_scenario_df()

    X_future, _ = generate_scenario(
        df,
        county_name="A COUNTY",
        state_abbr="CA",
        scenario_key="middle_road",
        horizon=2,
        baseline_yr=2023,
    )

    np.testing.assert_allclose(X_future["TAVG"].to_numpy(), np.array([13.0, 14.0]))
    np.testing.assert_allclose(X_future["HTDD"].to_numpy(), np.array([35.0, 30.0]))


def test_generate_scenario_raises_error_when_missing_baseline():
    df = make_scenario_df()

    with pytest.raises(ValueError, match="No data found"):
        generate_scenario(
            df,
            county_name="MISSING COUNTY",
            state_abbr="CA",
            scenario_key="middle_road",
            horizon=1,
            baseline_yr=2023,
        )
