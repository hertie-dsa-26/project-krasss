# scenarios.py
#
# This script generates synthetic future data for a given county under a
# predefined climate scenario. It is used by the Flask app to produce
# 10-year health outcome predictions when a user selects a county and scenario.
#
# This file defines:
#   SCENARIOS           — dictionary of available climate scenarios
#   generate_scenario() — builds synthetic future rows for a county
#
# How it works:
#   (1) Takes the 2023 row for the selected county as a baseline
#   (2) Creates horizon copies of that row — one per future year
#   (3) Applies the scenario delta to the relevant climate variable each year
#   (4) Drops all columns not used as features — same as prepare_data() in splitter.py
#   (5) Returns the future feature matrix ready for preprocessor.transform()
#
# Important: generate_scenario() returns raw unscaled features.
#            Always call preprocessor.transform() before passing to krr.predict()
#            Never call preprocessor.fit_transform() — the preprocessor must
#            already be fitted on 2013-2022 training data via load_model()
#
# Note on county uniqueness: county names are not unique across states.
#            For example CLINTON COUNTY exists in Michigan, Iowa and New York.
#            state_abbr is required to uniquely identify a county.

import numpy as np
import pandas as pd

# ── AVAILABLE SCENARIOS ───────────────────────────────────────────────────────

# Each scenario applies a multiplier to the historically observed trend per variable.
# A multiplier > 1 accelerates the trend, < 1 dampens it.
# Variables with no meaningful trend (e.g. PRCP) use a small noise factor instead.

SCENARIOS = {
    "low_warming": {
        "description": "Low warming — trends continue at a reduced rate (SSP1-2.6 analogue)",
        "trend_multiplier": 0.5,
    },
    "middle_road": {
        "description": "Middle-of-the-road — trends continue at historical rate (SSP2-4.5 analogue)",
        "trend_multiplier": 1.0,
    },
    "high_warming": {
        "description": "High warming — trends accelerate moderately (SSP3-7.0 analogue)",
        "trend_multiplier": 1.75,
    },
    "very_high_warming": {
        "description": "Very high warming — trends accelerate strongly (SSP5-8.5 analogue)",
        "trend_multiplier": 2.5,
    }
}

# Variables to apply trend projection to.
# Others are held constant at their 2023 baseline value.
TREND_VARS = ["TAVG", "TMAX", "TMIN", "CLDD", "HTDD", "DT100", "DX90", "EMXT", "EMNT"]
PRECIP_VARS = ["PRCP"]  # handled separately — no strong directional trend assumed
# Variables expected to INCREASE under warming — slope must be positive
WARMING_POSITIVE = ["TAVG", "TMAX", "TMIN", "CLDD", "DT100", "DX90", "EMXT"]
# Variables expected to DECREASE under warming — slope must be negative
WARMING_NEGATIVE = ["HTDD", "EMNT"]


def _compute_trend(df: pd.DataFrame, county_name: str, state_abbr: str, variable: str) -> float:
    """
    Fits a linear trend (slope per year) for a given variable in a given county
    using all available historical data.

    Returns:
        float: slope (change per year). Returns 0.0 if insufficient data.
    """
    county_df = df[
        (df["County name"] == county_name) &
        (df["StateAbbr"] == state_abbr) &
        df[variable].notna()
    ][["year", variable]].drop_duplicates().sort_values("year")

    if len(county_df) < 3:
        return 0.0

    x = county_df["year"].values
    y = county_df[variable].values
    slope = np.polyfit(x, y, 1)[0]
    
    # Enforce physically consistent direction for warming scenarios
    if variable in WARMING_POSITIVE:
        slope = abs(slope)       # must be positive (increasing)
    elif variable in WARMING_NEGATIVE:
        slope = -abs(slope)      # must be negative (decreasing)

    return float(slope)


def generate_scenario(df: pd.DataFrame, county_name: str, state_abbr: str,
                      scenario_key: str, horizon: int = 10, baseline_yr: int = 2023):
    """
    Builds synthetic future rows for a county under a given climate scenario.

    For each trend variable, fits a linear trend on all available historical data
    for that county, then projects forward by applying the scenario multiplier
    to that trend cumulatively each year.

    Args:
        df (pd.DataFrame):  Full dataset
        county_name (str):  Selected county
        state_abbr (str):   State abbreviation e.g. 'MI'
        scenario_key (str): One of 'low_warming', 'middle_road', 'high_warming', 'very_high_warming'
        horizon (int):      Number of future years — default 10
        baseline_yr (int):  Year to use as baseline — default 2023

    Returns:
        X_future (pd.DataFrame): Feature matrix ready for preprocessor.transform()
        future_years (list):     e.g. [2024, 2025, ..., 2033]
    """
    scenario = SCENARIOS[scenario_key]
    multiplier = scenario["trend_multiplier"]

    baseline = df[
        (df["County name"] == county_name) &
        (df["StateAbbr"] == state_abbr) &
        (df["year"] == baseline_yr)
    ].copy()

    if len(baseline) == 0:
        raise ValueError(f"No data found for {county_name}, {state_abbr} in {baseline_yr}.")
    if len(baseline) > 1:
        raise ValueError(f"Found {len(baseline)} rows for {county_name}, {state_abbr} in {baseline_yr}. Expected exactly 1.")

    # Pre-compute trends for all trend variables
    trends = {}
    for var in TREND_VARS:
        if var in df.columns:
            trends[var] = _compute_trend(df, county_name, state_abbr, var) * multiplier

    future_rows = []
    future_years = list(range(baseline_yr + 1, baseline_yr + 1 + horizon))

    for i, year in enumerate(future_years):
        row = baseline.copy()
        row["year"] = year

        # Apply projected trend cumulatively from baseline year
        for var, slope in trends.items():
            row[var] = baseline[var].values[0] + slope * (i + 1)

        future_rows.append(row)

    X_future = pd.concat(future_rows, ignore_index=True)

    drop_cols = ['year', 'StateAbbr', 'County name', 'CountyFIPS',
                 'STATION', 'STATION_NAME',
                 'BPHIGH', 'CASTHMA', 'COPD', 'MHLTH', 'PHLTH', 'SLEEP', 'STROKE']

    X_future = X_future.drop(columns=[c for c in drop_cols if c in X_future.columns])

    return X_future, future_years