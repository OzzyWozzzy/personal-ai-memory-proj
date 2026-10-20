"""Readiness covers the two local runtime dependencies only."""
from fastapi.testclient import TestClient

from app.engine import graph
from app.main import create_app


def test_readiness_checks_postgres_and_graph(monkeypatch, _database):
    monkeypatch.setattr(graph, "health_check", lambda: None)
    with TestClient(create_app()) as client:
        response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
    assert response.headers["cache-control"] == "no-store"


def test_readiness_failure_is_generic(monkeypatch, _database):
    def unavailable():
        raise RuntimeError("private graph detail")

    monkeypatch.setattr(graph, "health_check", unavailable)
    with TestClient(create_app()) as client:
        response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json() == {"status": "not ready"}
    assert "private graph detail" not in response.text
