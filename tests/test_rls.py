"""Row-level security and grant enforcement.

Proves the three isolation properties the rest of the suite relies on:
cross-tenant reads return nothing, cross-tenant writes are rejected by the
policy's WITH CHECK, and audit_log is append-only for the app role.
"""
from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from app.audit import write_audit


def test_cross_tenant_read_returns_zero_rows(
    make_tenant, tenant_conn, make_entity, make_episode
):
    tenant_a = make_tenant()
    tenant_b = make_tenant()

    with tenant_conn(tenant_a.tenant_id) as conn:
        episode_id = make_episode(conn, tenant_a.tenant_id, content="tenant A episode")
        entity_id = make_entity(conn, tenant_a.tenant_id, "Alice")

    # Tenant A sees its own rows...
    with tenant_conn(tenant_a.tenant_id) as conn:
        assert (
            conn.execute(
                text("SELECT count(*) FROM episodes WHERE episode_id = :e"),
                {"e": str(episode_id)},
            ).scalar()
            == 1
        )

    # ...tenant B sees none of them.
    with tenant_conn(tenant_b.tenant_id) as conn:
        n_episodes = conn.execute(text("SELECT count(*) FROM episodes")).scalar()
        n_entities = conn.execute(text("SELECT count(*) FROM entities")).scalar()
        n_specific = conn.execute(
            text("SELECT count(*) FROM entities WHERE entity_id = :e"),
            {"e": str(entity_id)},
        ).scalar()
    assert n_episodes == 0
    assert n_entities == 0
    assert n_specific == 0


def test_insert_with_mismatched_tenant_id_rejected(make_tenant, tenant_conn):
    tenant_a = make_tenant()
    tenant_b = make_tenant()

    # Connection carries tenant A's context but the row claims tenant B:
    # the RLS policy's WITH CHECK must reject it.
    with pytest.raises(DBAPIError) as excinfo:
        with tenant_conn(tenant_a.tenant_id) as conn:
            conn.execute(
                text(
                    "INSERT INTO entities (tenant_id, entity_type, canonical_name) "
                    "VALUES (:tid, 'person', 'Mallory')"
                ),
                {"tid": str(tenant_b.tenant_id)},
            )
    assert "row-level security" in str(excinfo.value)


def test_audit_log_is_append_only_for_app_role(make_tenant, tenant_conn):
    tenant = make_tenant()

    with tenant_conn(tenant.tenant_id) as conn:
        write_audit(conn, tenant.tenant_id, "api:test", "fact.insert", {"why": "test"})

    with tenant_conn(tenant.tenant_id) as conn:
        assert conn.execute(text("SELECT count(*) FROM audit_log")).scalar() == 1

    with pytest.raises(DBAPIError) as excinfo:
        with tenant_conn(tenant.tenant_id) as conn:
            conn.execute(text("UPDATE audit_log SET action = 'tampered'"))
    assert "permission denied" in str(excinfo.value)
