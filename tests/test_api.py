from fastapi.testclient import TestClient

from src.api.app import app


client = TestClient(app)

def test_root_endpoint():

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Customer Churn API is running"


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_docs_available():

    response = client.get("/docs")

    assert response.status_code == 200