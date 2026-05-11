import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[4] / "app"
sys.path.insert(0, str(APP_DIR))

from main import app



# Testing whether '/api/timeseries' works:

def test_timeseries_api_returns_success():
    client = app.test_client()

    response = client.get("/api/timeseries?health=MHLTH&weather=TAVG&state=all")

    assert response.status_code == 200


def test_timeseries_api_returns_health_and_weather_series():
    client = app.test_client()

    response = client.get("/api/timeseries?health=MHLTH&weather=TAVG&state=all")
    data = response.get_json()

    assert data["state"] == "all"
    assert "data" in data
    assert "health" in data["data"]
    assert "weather" in data["data"]


def test_timeseries_api_series_contains_expected_keys():
    client = app.test_client()

    response = client.get("/api/timeseries?health=MHLTH&weather=TAVG&state=all")
    data = response.get_json()

    health_data = data["data"]["health"]

    assert "series" in health_data
    assert "national" in health_data
    assert "var" in health_data
    assert "stats" in health_data
    assert health_data["var"] == "MHLTH"


def test_timeseries_heatmap_api_returns_success():
    client = app.test_client()

    response = client.get("/api/timeseries/heatmap?var=MHLTH&states=CA,NY")

    assert response.status_code == 200


def test_timeseries_heatmap_api_returns_data_for_requested_states():
    client = app.test_client()

    response = client.get("/api/timeseries/heatmap?var=MHLTH&states=CA,NY")
    data = response.get_json()

    assert data["variable"] == "MHLTH"
    assert "data" in data
    assert "CA" in data["data"]
    assert "NY" in data["data"]


def test_timeseries_heatmap_api_returns_400_for_invalid_var():
    client = app.test_client()

    response = client.get("/api/timeseries/heatmap?var=NOT_A_COLUMN")
    data = response.get_json()

    assert response.status_code == 400
    assert data == {"error": "Invalid variable"}