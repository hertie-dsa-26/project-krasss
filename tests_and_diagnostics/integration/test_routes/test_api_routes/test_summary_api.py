import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[4] / "app"
sys.path.insert(0, str(APP_DIR))

from main import app


# Testing whether '/api/summary' works:

def test_summary_api_returns_stats_for_valid_col():
    client = app.test_client()

    response = client.post("/api/summary", json={"column": "MHLTH"})
    data = response.get_json()

    assert response.status_code == 200
    assert data["column"] == "MHLTH"
    assert "stats" in data
    assert "mean" in data["stats"]


def test_summary_api_returns_400_for_invalid_col():
    client = app.test_client()

    response = client.post("/api/summary", json={"column": "NOT_A_COLUMN"})
    data = response.get_json()

    assert response.status_code == 400
    assert data == {"error": "Invalid column name"}