from fastapi.testclient import TestClient

from second_brain.web import app


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "service": "second-brain"}


def test_home_page_is_served() -> None:
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert "Second Brain" in response.text
