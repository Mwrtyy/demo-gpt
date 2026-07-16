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


def test_zero_lab_page_and_status_are_served() -> None:
    client = TestClient(app)
    page = client.get("/zero")
    status = client.get("/api/zero/status")

    assert page.status_code == 200
    assert "Second Brain Zero Lab" in page.text
    assert status.status_code == 200
    payload = status.json()
    assert "checkpoint_present" in payload
    assert "dependencies_available" in payload
    assert "ready" in payload
