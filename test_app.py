
import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "healthy"}


def test_prediction_success(client):
    payload = {
        "MedInc": 5.0,
        "HouseAge": 20.0,
        "AveRooms": 5.0,
        "AveBedrms": 1.0,
        "Population": 1000.0,
        "AveOccup": 3.0,
        "Latitude": 34.0,
        "Longitude": -118.0,
    }

    response = client.post("/prediction", json=payload)

    assert response.status_code == 200

    result = response.get_json()
    assert "predicted_house_value" in result
    assert result["currency"] == "INR"
    assert isinstance(result["predicted_house_value"], (int, float))


def test_prediction_missing_features(client):
    response = client.post(
        "/prediction",
        json={"MedInc": 5.0},
    )

    assert response.status_code == 400
    assert "missing_features" in response.get_json()


def test_prediction_invalid_input(client):
    payload = {
        "MedInc": "invalid",
        "HouseAge": 20.0,
        "AveRooms": 5.0,
        "AveBedrms": 1.0,
        "Population": 1000.0,
        "AveOccup": 3.0,
        "Latitude": 34.0,
        "Longitude": -118.0,
    }

    response = client.post("/prediction", json=payload)

    assert response.status_code == 400
