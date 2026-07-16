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
    assert "One-click autonomous training" in page.text
    assert status.status_code == 200
    payload = status.json()
    assert "checkpoint_present" in payload
    assert "dependencies_available" in payload
    assert "ready" in payload


def test_training_catalog_and_idle_status_are_served() -> None:
    client = TestClient(app)
    catalog = client.get("/api/zero/training/catalog")
    status = client.get("/api/zero/training/status")

    assert catalog.status_code == 200
    generations = catalog.json()["generations"]
    level1 = next(item for item in generations if item["id"] == "level1")
    level2 = next(item for item in generations if item["id"] == "level2")
    assert level1["parameters"] == 19_143_168
    assert level2["parameters"] == 38_023_680
    assert status.status_code == 200
    assert "status" in status.json()
    assert "active" in status.json()
