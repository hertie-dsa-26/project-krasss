import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[4] / "app"
sys.path.insert(0, str(APP_DIR))

from main import app



# Testing whether '/api/snapshot' works:

def test_snapshot_api_returns_success():
    client = app.test_client()

    response = client.get("/api/snapshot")

    assert response.status_code == 200


def test_snapshot_api_returns_expected_keys():
    client = app.test_client()

    response = client.get("/api/snapshot")
    data = response.get_json()

    assert set(data.keys()) == {"states", "counties", "years", "n_vars"}


def test_snapshot_api_returns_positive_counts(): # i.e., nonempty and sensible values.
    client = app.test_client()

    response = client.get("/api/snapshot")
    data = response.get_json()

    assert data["states"] > 0
    assert data["counties"] > 0
    assert data["years"] > 0
    assert data["n_vars"] > 0
