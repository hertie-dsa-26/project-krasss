import sys
from pathlib import Path

import pandas as pd

APP_FUNCTIONS = Path(__file__).resolve().parents[3] / "app" / "functions"
sys.path.insert(0, str(APP_FUNCTIONS))

from preprocessing import detect_categorical_columns



def test_detect_categorical_columns_returns_indices():
    X_df = pd.DataFrame({
        "temperature": [10.0, 20.0, 30.0],
        "climate_type": ["dry", "dry", "wet"],
        "population": [100, 200, 300],
    })

    result = detect_categorical_columns(X_df)

    assert result == [1]


def test_detect_categorical_columns_returns_multiple_indices():
    X_df = pd.DataFrame({
        "temperature": [10.0, 20.0, 30.0],
        "climate_type": ["dry", "dry", "wet"],
        "region": pd.Series(["A", "B", "A"], dtype="category"),
        "population": [100, 200, 300],
    })

    result = detect_categorical_columns(X_df)

    assert result == [1, 2]


def test_detect_categorical_columns_returns_empty_list_when_no_cat():
    X_df = pd.DataFrame({
        "temperature": [10.0, 20.0, 30.0],
        "population": [100, 200, 300],
    })

    result = detect_categorical_columns(X_df)

    assert result == []
