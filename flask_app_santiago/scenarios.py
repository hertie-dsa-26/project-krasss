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

# Each scenario defines which climate variable changes and by how much per year.
# All other features are held constant at their 2023 values for the selected county.
# Add new scenarios here — the Flask app will pick them up automatically.
SCENARIOS = {
    "tavg_increase_0.1": {
        "description": "TAVG increases by 0.1°C per year",
        "variable":     "TAVG",
        "delta_per_year": 0.1
    },
    "tavg_increase_0.5": {
        "description": "TAVG increases by 0.5°C per year",
        "variable":     "TAVG",
        "delta_per_year": 0.5
    }
    # add more scenarios here
}


def generate_scenario(df: pd.DataFrame, county_name: str, state_abbr: str,
                      scenario_key: str, horizon: int = 10):
    """
    Builds synthetic future rows for a county under a given climate scenario.

    Takes the 2023 row for the selected county as a baseline and creates
    one row per future year, applying the scenario delta cumulatively.
    All features other than the scenario variable are held constant at
    their 2023 values.

    Args:
        df (pd.DataFrame):  Full dataset — needed to find the 2023 baseline row
        county_name (str):  Selected county — must match exactly as it appears in the dataset
        state_abbr (str):   State abbreviation e.g. 'MI' — needed because county
                            names are not unique across states
        scenario_key (str): Key from the SCENARIOS dict e.g. 'tavg_increase_0.1'
        horizon (int):      Number of future years to generate — default 10

    Returns:
        X_future (pd.DataFrame): Feature matrix with horizon rows ready for
                                 preprocessor.transform() — NOT yet scaled
        future_years (list):     Corresponding years e.g. [2024, 2025, ..., 2033]
    """
    scenario = SCENARIOS[scenario_key]

    # Filter by both county name and state to get a unique row.
    # County names alone are not unique — e.g. CLINTON COUNTY exists in
    # multiple states. StateAbbr ensures we get exactly one baseline row.
    baseline = df[
        (df["County name"] == county_name) &
        (df["StateAbbr"] == state_abbr) &
        (df["year"] == 2023)
    ].copy()

    assert len(baseline) == 1, \
        f"Expected exactly 1 row for {county_name}, {state_abbr} in 2023, found {len(baseline)}"

    future_rows = []
    future_years = list(range(2024, 2024 + horizon))

    for i, year in enumerate(future_years):
        row = baseline.copy()
        row["year"] = year

        # Apply the scenario delta cumulatively
        # Year 1 gets +0.1, year 2 gets +0.2, year 3 gets +0.3 etc.
        row[scenario["variable"]] += scenario["delta_per_year"] * (i + 1)

        future_rows.append(row)

    X_future = pd.concat(future_rows, ignore_index=True)

    # Drop identifier columns and health variables — must match prepare_data() in splitter.py
    # What remains are the 28 feature columns that the model was trained on
    drop_cols = ['year', 'StateAbbr', 'County name', 'CountyFIPS',
                 'STATION', 'STATION_NAME',
                 'BPHIGH', 'CASTHMA', 'COPD', 'MHLTH', 'PHLTH', 'SLEEP', 'STROKE']

    X_future = X_future.drop(columns=drop_cols)

    return X_future, future_years
