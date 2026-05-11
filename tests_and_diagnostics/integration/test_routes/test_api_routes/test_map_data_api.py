import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[4] / "app"
sys.path.insert(0, str(APP_DIR))

from main import app



# Testing whether '/api/map-data' works:

def test_map_data_api_returns_success_for_valid_health_var():
    client = app.test_client()

    response = client.get("/api/map-data?var=MHLTH")

    assert response.status_code == 200


def test_map_data_api_returns_expected_top_level_keys():
    client = app.test_client()

    response = client.get("/api/map-data?var=MHLTH")
    data = response.get_json()

    expected_keys = {
        "data",
        "health_nat_avg",
        "weather_nat_avg",
        "health_var",
        "weather_var",
        "demo_vars",
        "year_start",
        "year_end",
    }

    assert set(data.keys()) == expected_keys


def test_map_data_api_returns_county_records():
    client = app.test_client()

    response = client.get("/api/map-data?var=MHLTH")
    data = response.get_json()

    assert len(data["data"]) > 0

    first_record = data["data"][0]

    assert "fips" in first_record
    assert "county" in first_record
    assert "state" in first_record
    assert "health_val" in first_record
    assert "population" in first_record


def test_map_data_api_returns_400_for_invalid_health_var():
    client = app.test_client()

    response = client.get("/api/map-data?var=NOT_A_COLUMN")
    data = response.get_json()

    assert response.status_code == 400
    assert data == {"error": "Invalid variable"}