"""Focused local HTTP boundary tests."""
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.security import SecurityConfig, local_preview_enabled


def test_local_preview_defaults_on_and_can_be_disabled(monkeypatch):
    monkeypatch.delenv("MEMORY_ENABLE_LOCAL_PREVIEW", raising=False)
    assert local_preview_enabled() is True
    monkeypatch.setenv("MEMORY_ENABLE_LOCAL_PREVIEW", "0")
    assert local_preview_enabled() is False


def test_security_config_rejects_wildcards_and_bad_size(monkeypatch):
    monkeypatch.setenv("MEMORY_ALLOWED_HOSTS", "*")
    with pytest.raises(RuntimeError, match="explicit hosts"):
        SecurityConfig.from_env()
    monkeypatch.setenv("MEMORY_ALLOWED_HOSTS", "localhost")
    monkeypatch.setenv("MEMORY_MAX_REQUEST_BYTES", "zero")
    with pytest.raises(RuntimeError, match="integer"):
        SecurityConfig.from_env()


def test_security_headers_and_host_boundary(monkeypatch):
    monkeypatch.delenv("MEMORY_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("MEMORY_ALLOWED_ORIGINS", raising=False)
    client = TestClient(create_app())
    response = client.get("/healthz")
    assert response.status_code == 200
    assert "default-src 'self'" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert client.get("/healthz", headers={"Host": "evil.example"}).status_code == 400


def test_cross_origin_is_rejected():
    response = TestClient(create_app()).get(
        "/healthz", headers={"Origin": "https://evil.example"}
    )
    assert response.status_code == 403


def test_declared_oversized_body_is_rejected(monkeypatch):
    monkeypatch.setenv("MEMORY_MAX_REQUEST_BYTES", "16")
    response = TestClient(create_app()).post(
        "/v1/remember",
        headers={"X-API-Key": "unknown", "Content-Type": "application/json"},
        content='{"text":"this payload is too large"}',
    )
    assert response.status_code == 413
