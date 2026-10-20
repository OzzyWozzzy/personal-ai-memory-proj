"""Hash-only agent-key lifecycle, capabilities, and tenant isolation."""
from __future__ import annotations

import hashlib
import json

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from app import db
from app.api import agent_access


def h(key_or_tenant):
    key = getattr(key_or_tenant, "api_key", key_or_tenant)
    return {"X-API-Key": key}


def test_create_returns_one_time_high_entropy_key(make_tenant, client):
    tenant = make_tenant()
    response = client.post(
        "/v1/agent_keys",
        headers=h(tenant),
        json={"label": "Claude Desktop", "expires_in_days": 30},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["label"] == "Claude Desktop"
    assert body["api_key"].startswith("mem_agent_")
    assert len(body["api_key"]) >= 50
    assert body["key_prefix"] == body["api_key"][:18]
    assert body["key_preview"] == body["key_prefix"] + "…"
    assert body["key_id"]
    assert body["scopes"] == ["memory:read", "memory:write"]
    assert body["expires_at"]
    assert "won't be shown again" in body["note"]


def test_database_stores_digest_not_raw_key(make_tenant, client, admin_engine):
    tenant = make_tenant()
    created = client.post(
        "/v1/agent_keys", headers=h(tenant), json={"label": "Hash only"}
    ).json()
    with admin_engine.connect() as conn:
        row = conn.execute(
            text("SELECT key_hash, key_prefix FROM api_keys WHERE key_id = :key_id"),
            {"key_id": created["key_id"]},
        ).one()
        columns = {
            result.column_name
            for result in conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema='public' AND table_name='api_keys'"
                )
            )
        }
        obsolete_auth_tables = conn.execute(
            text(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_schema='public' "
                "AND table_name IN "
                "('accounts', 'account_sessions', 'account_projects')"
            )
        ).scalar_one()
    assert bytes(row.key_hash) == hashlib.sha256(
        created["api_key"].encode("utf-8")
    ).digest()
    assert row.key_prefix == created["key_prefix"]
    assert "api_key" not in columns
    assert obsolete_auth_tables == 0


def test_expired_key_is_rejected_and_last_used_is_tracked(
    make_tenant, client, admin_engine
):
    tenant = make_tenant()
    created = client.post(
        "/v1/agent_keys", headers=h(tenant), json={"label": "Expiring"}
    ).json()
    assert db.resolve_api_key(created["api_key"]) is not None
    with admin_engine.begin() as conn:
        assert conn.execute(
            text("SELECT last_used_at IS NOT NULL FROM api_keys WHERE key_id=:key_id"),
            {"key_id": created["key_id"]},
        ).scalar_one() is True
        conn.execute(
            text(
                "UPDATE api_keys SET expires_at = now() - interval '1 second' "
                "WHERE key_id = :key_id"
            ),
            {"key_id": created["key_id"]},
        )
    assert db.resolve_api_key(created["api_key"]) is None


def test_list_shows_metadata_never_key_or_hash(make_tenant, client):
    tenant = make_tenant()
    first = client.post(
        "/v1/agent_keys", headers=h(tenant), json={"label": "Bot A"}
    ).json()
    second = client.post(
        "/v1/agent_keys", headers=h(tenant), json={"label": "Bot B"}
    ).json()

    response = client.get("/v1/agent_keys", headers=h(tenant))
    assert response.status_code == 200
    keys = response.json()["keys"]
    assert [item["label"] for item in keys] == ["Bot A", "Bot B"]
    assert {item["key_id"] for item in keys} == {
        first["key_id"], second["key_id"]
    }
    assert {item["key_preview"] for item in keys} == {
        first["key_preview"], second["key_preview"]
    }
    assert all(item["active"] is True for item in keys)
    assert all(item["created_at"] for item in keys)
    blob = json.dumps(response.json())
    assert first["api_key"] not in blob and second["api_key"] not in blob
    assert "key_hash" not in blob


def test_agent_key_works_then_revoke_by_key_id_is_immediate(make_tenant, client):
    tenant = make_tenant()
    created = client.post(
        "/v1/agent_keys", headers=h(tenant), json={"label": "Doomed"}
    ).json()
    api_key = created["api_key"]

    assert client.get("/v1/stats", headers=h(api_key)).status_code == 200
    assert db.resolve_api_key(api_key) is not None

    response = client.post(
        "/v1/agent_keys/revoke",
        headers=h(tenant),
        json={"key_id": created["key_id"]},
    )
    assert response.status_code == 200, response.text
    assert response.json()["revoked"] is True
    assert response.json()["key_id"] == created["key_id"]

    assert db.resolve_api_key(api_key) is None
    assert client.get("/v1/stats", headers=h(api_key)).status_code == 401
    listed = client.get("/v1/agent_keys", headers=h(tenant)).json()["keys"]
    assert [item["active"] for item in listed] == [False]


def test_revoke_by_listed_prefix(make_tenant, client):
    tenant = make_tenant()
    created = client.post(
        "/v1/agent_keys", headers=h(tenant), json={"label": "Prefix"}
    ).json()
    response = client.post(
        "/v1/agent_keys/revoke",
        headers=h(tenant),
        json={"key_prefix": created["key_preview"]},
    )
    assert response.status_code == 200, response.text
    assert response.json()["key_prefix"] == created["key_prefix"]
    assert db.resolve_api_key(created["api_key"]) is None


def test_revoke_ambiguous_prefix_requires_key_id(
    make_tenant, client, monkeypatch
):
    tenant = make_tenant()
    nonce = str(tenant.tenant_id)
    values = iter((f"abcdefgh-one-{nonce}", f"abcdefgh-two-{nonce}"))
    monkeypatch.setattr(agent_access.secrets, "token_urlsafe", lambda _: next(values))
    first = client.post(
        "/v1/agent_keys", headers=h(tenant), json={"label": "A"}
    ).json()
    second = client.post(
        "/v1/agent_keys", headers=h(tenant), json={"label": "B"}
    ).json()
    assert first["key_prefix"] == second["key_prefix"]

    response = client.post(
        "/v1/agent_keys/revoke",
        headers=h(tenant),
        json={"key_prefix": first["key_prefix"]},
    )
    assert response.status_code == 409
    assert "key_id" in response.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    ({}, {"key_prefix": "…"}, {"key_prefix": "xx"}),
)
def test_revoke_requires_one_safe_selector(make_tenant, client, payload):
    tenant = make_tenant()
    response = client.post(
        "/v1/agent_keys/revoke", headers=h(tenant), json=payload
    )
    assert response.status_code == 422


def test_revoke_rejects_both_selectors(make_tenant, client):
    tenant = make_tenant()
    created = client.post(
        "/v1/agent_keys", headers=h(tenant), json={"label": "Both"}
    ).json()
    response = client.post(
        "/v1/agent_keys/revoke",
        headers=h(tenant),
        json={
            "key_id": created["key_id"],
            "key_prefix": created["key_prefix"],
        },
    )
    assert response.status_code == 422


def test_keys_are_tenant_scoped_by_force_rls(make_tenant, client, tenant_conn):
    tenant_a, tenant_b = make_tenant(), make_tenant()
    created = client.post(
        "/v1/agent_keys", headers=h(tenant_a), json={"label": "A bot"}
    ).json()

    assert client.get("/v1/agent_keys", headers=h(tenant_b)).json()["keys"] == []
    response = client.post(
        "/v1/agent_keys/revoke",
        headers=h(tenant_b),
        json={"key_id": created["key_id"]},
    )
    assert response.status_code == 404
    assert db.resolve_api_key(created["api_key"]) is not None

    with tenant_conn(tenant_b.tenant_id) as conn:
        assert conn.execute(text("SELECT count(*) FROM api_keys")).scalar_one() == 1
        # The one visible row is tenant_b's root fixture, never tenant_a's key.
        assert conn.execute(
            text("SELECT count(*) FROM api_keys WHERE key_id = :key_id"),
            {"key_id": created["key_id"]},
        ).scalar_one() == 0


def test_app_role_cannot_read_key_hash(make_tenant, tenant_conn):
    tenant = make_tenant()
    with pytest.raises(DBAPIError) as excinfo:
        with tenant_conn(tenant.tenant_id) as conn:
            conn.execute(text("SELECT key_hash FROM api_keys")).all()
    assert "permission denied" in str(excinfo.value)


def test_agent_capabilities_cannot_mint_admin_keys(
    make_tenant, client
):
    tenant = make_tenant()
    agent_key = client.post(
        "/v1/agent_keys", headers=h(tenant), json={"label": "Limited"}
    ).json()["api_key"]

    assert client.get("/v1/stats", headers=h(agent_key)).status_code == 200
    assert client.post(
        "/v1/agent_keys", headers=h(agent_key), json={"label": "Escalation"}
    ).status_code == 403
