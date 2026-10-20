"""The local UI workspace persists while its HttpOnly credential rotates."""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from sqlalchemy import text

from app import db
from app.main import create_app
from app.security import DEV_PREVIEW_COOKIE


ORIGIN = {"Origin": "http://testserver"}


def test_local_preview_is_persistent_httponly_and_rotates_keys(client, monkeypatch):
    monkeypatch.setenv("MEMORY_AUTH_MODE", "development")
    monkeypatch.setenv("MEMORY_ENABLE_LOCAL_PREVIEW", "1")
    # Pin the workspace tenant per-test so reruns against a session-scoped
    # database stay isolated; this also exercises the env override that lets a
    # deployment share the demo workspace with an MCP client's key.
    workspace = uuid.uuid4()
    monkeypatch.setenv("MEMORY_PREVIEW_TENANT_ID", str(workspace))
    # The session-scoped client was built before the flag changed; use an app
    # whose route table reflects the explicit preview setting.
    with TestClient(create_app()) as preview_client:
        capability = preview_client.get("/v1/auth/preview")
        assert capability.status_code == 200
        assert capability.json() == {"enabled": True}

        assert preview_client.post("/v1/auth/preview").status_code == 403
        started = preview_client.post("/v1/auth/preview", headers=ORIGIN)
        assert started.status_code == 201, started.text
        assert "api_key" not in started.text
        assert "mem_preview_" not in started.text
        workspace_info = started.json()["workspace"]
        assert workspace_info["id"] == str(workspace)
        assert workspace_info["name"] == "Local Memory"
        # A fresh workspace starts EMPTY — nothing is seeded.
        assert workspace_info["counts"] == {"episodes": 0, "facts": 0}
        cookies = started.headers.get_list("set-cookie")
        assert any(DEV_PREVIEW_COOKIE in value and "HttpOnly" in value
                   and "SameSite=strict" in value for value in cookies)

        # Model a stored source without coupling this credential test to LLM
        # extraction; /v1/remember owns that integration contract elsewhere.
        with db.get_admin_engine().begin() as conn:
            conn.execute(text(
                "INSERT INTO episodes (tenant_id, source, occurred_at, content,"
                " content_hash) VALUES (:t, 'chat', now(), :content, :hash)"
            ), {"t": str(workspace), "content": "The workspace remembers this.",
                "hash": uuid.uuid4().hex})

        # The demo may mint the same narrow key a user copies into an MCP
        # client; it inherits the preview lifetime.
        created = preview_client.post(
            "/v1/agent_keys",
            headers=ORIGIN,
            json={"label": "ChatGPT MCP demo"},
        )
        assert created.status_code == 201, created.text
        agent_key = created.json()["api_key"]
        assert created.json()["scopes"] == ["memory:read", "memory:write"]
        agent_expires_at = datetime.fromisoformat(created.json()["expires_at"])
        assert datetime.now(timezone.utc) < agent_expires_at
        assert agent_expires_at <= datetime.now(timezone.utc) + timedelta(hours=8)
        assert preview_client.get(
            "/v1/admin/token_stats"
        ).status_code == 200

        raw_key = preview_client.cookies.get(DEV_PREVIEW_COOKIE)
        assert raw_key
        with TestClient(create_app()) as agent_client:
            agent_headers = {"X-API-Key": agent_key}
            stats = agent_client.get("/v1/stats", headers=agent_headers)
            assert stats.status_code == 200, stats.text
            assert stats.json()["episodes"] == 1

            stopped = preview_client.delete("/v1/auth/preview", headers=ORIGIN)
            assert stopped.status_code == 200
            assert stopped.headers["clear-site-data"] == (
                '"cache", "cookies", "storage"'
            )
            # The workspace is persistent: exiting the preview revokes the
            # PREVIEW credential but not the user-managed (and self-expiring)
            # agent key — an MCP assistant keeps working.
            assert agent_client.get(
                "/v1/stats", headers=agent_headers
            ).status_code == 200

        # The preview cookie credential itself is dead...
        assert preview_client.get("/v1/graph").status_code == 401
        with db.get_admin_engine().connect() as conn:
            active = conn.execute(
                text("SELECT active FROM api_keys WHERE key_hash=:key_hash"),
                {"key_hash": hashlib.sha256(raw_key.encode()).digest()},
            ).scalar_one()
        assert active is False

        # ...and re-entering the demo lands in the SAME workspace with the
        # earlier entry still there.
        again = preview_client.post("/v1/auth/preview", headers=ORIGIN)
        assert again.status_code == 201, again.text
        workspace2 = again.json()["workspace"]
        assert workspace2["id"] == workspace_info["id"]
        assert workspace2["counts"]["episodes"] == 1


def test_local_preview_route_is_absent_without_flag(monkeypatch):
    monkeypatch.setenv("MEMORY_AUTH_MODE", "development")
    monkeypatch.setenv("MEMORY_ENABLE_LOCAL_PREVIEW", "0")
    with TestClient(create_app()) as client:
        assert client.get("/v1/auth/preview").status_code == 404
