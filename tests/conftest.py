"""Shared fixtures for the temporal-memory test suite.

Fixture names here are a cross-team contract — other workstreams' tests import
them exactly as named. Tenancy is the isolation mechanism: every make_tenant()
call mints a fresh random tenant, so RLS keeps tests from seeing each other and
no teardown is needed.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import text

# Unit and integration tests must never inherit developer secrets or a live LLM
# mode merely because gitignored dotenv files exist beside the checkout.
os.environ["MEMORY_LOAD_LOCAL_ENV"] = "0"
# Never run integration fixtures against the persistent local application DB.
# This must be selected before importing app.db because tests import application
# modules during collection, before session fixtures execute.
os.environ.setdefault(
    "MEMORY_DEVDB_DIR", tempfile.mkdtemp(prefix="temporal-memory-pytest-")
)

from app import db
from app.main import create_app

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session", autouse=True)
def _database(tmp_path_factory):
    """Boot isolated embedded databases and migrate Postgres once per session.

    Both Postgres fixture rows and embedded FalkorDB data land in temporary
    directories, never the persistent local application stores.
    """
    os.environ.setdefault("MEMORY_GRAPH_DIR", str(tmp_path_factory.mktemp("graphdb")))
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    command.upgrade(cfg, "head")
    yield
    # Deterministically stop any embedded falkordblite server the session
    # booted (per-file fixtures close their own roots; this covers the rest).
    from app.engine import graph as _graph

    _graph.close_all()


@pytest.fixture(scope="session")
def admin_engine(_database):
    return db.get_admin_engine()


@pytest.fixture(scope="session")
def app_engine(_database):
    return db.get_app_engine()


@pytest.fixture
def make_tenant(admin_engine):
    """Factory: fresh tenant with api_keys + tenant_settings rows. Returns
    SimpleNamespace(tenant_id, api_key). v4 has no relation registry to seed —
    entities/edges live in the per-tenant graph, projected on ingest."""

    def _make_tenant() -> SimpleNamespace:
        tenant_id = uuid.uuid4()
        api_key = f"key-{tenant_id}"
        key_hash = hashlib.sha256(api_key.encode("utf-8")).digest()
        with admin_engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO api_keys "
                    "(key_hash, key_prefix, tenant_id, name, scopes) "
                    "VALUES (:key_hash, :key_prefix, :tid, 'test', "
                    "ARRAY['memory:read', 'memory:write', "
                    "      'workspace:admin']::text[])"
                ),
                {
                    "key_hash": key_hash,
                    "key_prefix": api_key[:12],
                    "tid": str(tenant_id),
                },
            )
            conn.execute(
                text(
                    "INSERT INTO tenant_settings (tenant_id, timezone, locale) "
                    "VALUES (:tid, 'UTC', 'en')"
                ),
                {"tid": str(tenant_id)},
            )
        return SimpleNamespace(tenant_id=tenant_id, api_key=api_key)

    return _make_tenant


@pytest.fixture
def tenant_conn(_database):
    """Factory: tenant_conn(tenant_id) -> contextmanager yielding an app-role
    connection inside a transaction with app.tenant_id set."""

    def _tenant_conn(tenant_id, isolation_level: str | None = None):
        return db.tenant_connection(tenant_id, isolation_level=isolation_level)

    return _tenant_conn


@pytest.fixture
def client(_database):
    return TestClient(create_app())


def _make_entity(conn, tenant_id, name, entity_type="person", identifiers=None):
    row = conn.execute(
        text(
            """
            INSERT INTO entities (tenant_id, entity_type, canonical_name, identifiers)
            VALUES (:tid, :etype, :name, CAST(:idents AS jsonb))
            RETURNING entity_id
            """
        ),
        {
            "tid": str(tenant_id),
            "etype": entity_type,
            "name": name,
            "idents": json.dumps(identifiers or {}),
        },
    ).one()
    return row.entity_id


def _make_episode(
    conn,
    tenant_id,
    source="manual_fact",
    content="",
    occurred_at=None,
    content_json=None,
):
    # content_hash is the idempotency key (UNIQUE per tenant): hash the caller's
    # content when given so re-ingestion semantics hold; otherwise a random
    # nonce keeps every helper-made episode unique.
    if content:
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    else:
        content_hash = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
    row = conn.execute(
        text(
            """
            INSERT INTO episodes
              (tenant_id, source, occurred_at, content, content_json, content_hash)
            VALUES
              (:tid, :source, :occurred_at, :content, CAST(:cjson AS jsonb), :chash)
            RETURNING episode_id
            """
        ),
        {
            "tid": str(tenant_id),
            "source": source,
            "occurred_at": occurred_at or datetime.now(timezone.utc),
            "content": content,
            "cjson": json.dumps(content_json) if content_json is not None else None,
            "chash": content_hash,
        },
    ).one()
    return row.episode_id


@pytest.fixture
def make_entity():
    """make_entity(conn, tenant_id, name, entity_type='person', identifiers=None) -> UUID"""
    return _make_entity


@pytest.fixture
def make_episode():
    """make_episode(conn, tenant_id, source='manual_fact', content='',
    occurred_at=None, content_json=None) -> UUID"""
    return _make_episode
