import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[3] / "app"
sys.path.insert(0, str(APP_DIR))

from main import app



# Testing whether the main Flask pages load successfully:

def test_home_route_returns_success():
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200


def test_docs_route_returns_success():
    client = app.test_client()

    response = client.get("/docs")

    assert response.status_code == 200


def test_explore_route_returns_success():
    client = app.test_client()

    response = client.get("/explore")

    assert response.status_code == 200


def test_predict_get_route_returns_success():
    client = app.test_client()

    response = client.get("/predict")

    assert response.status_code == 200