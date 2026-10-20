"""Postgres engines, workspace isolation, and hash-only API-key resolution."""
from __future__ import annotations

import hashlib
import os
import threading
from contextlib import contextmanager
from types import SimpleNamespace

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app import devdb

_admin_engine: Engine | None = None
_app_engine: Engine | None = None
_engine_lock = threading.Lock()


class InactiveTenantError(PermissionError):
    """The selected local workspace is inactive or absent."""


def get_admin_url() -> str:
    return os.environ.get("POSTGRES_ADMIN_URL") or devdb.ensure_devdb()[0]


def get_app_url() -> str:
    return os.environ.get("POSTGRES_APP_URL") or devdb.ensure_devdb()[1]


def _engine(url: str, *, admin: bool) -> Engine:
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_use_lifo=True,
        pool_size=1 if admin else 3,
        max_overflow=1 if admin else 2,
        pool_timeout=10,
        pool_recycle=300,
        hide_parameters=True,
    )


def get_admin_engine() -> Engine:
    global _admin_engine
    if _admin_engine is None:
        with _engine_lock:
            if _admin_engine is None:
                _admin_engine = _engine(get_admin_url(), admin=True)
    return _admin_engine


def get_app_engine() -> Engine:
    global _app_engine
    if _app_engine is None:
        with _engine_lock:
            if _app_engine is None:
                _app_engine = _engine(get_app_url(), admin=False)
    return _app_engine


def _lock_id(tenant_id, namespace: str = "workspace") -> int:
    digest = hashlib.sha256(f"{namespace}:{tenant_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


def _set_tenant_context(conn, tenant_id, *, require_active: bool) -> None:
    conn.execute(
        text("SELECT set_config('app.tenant_id', :tenant_id, true)"),
        {"tenant_id": str(tenant_id)},
    )
    if require_active:
        active = conn.execute(
            text(
                "SELECT active FROM tenant_settings "
                "WHERE tenant_id=:tenant_id AND active"
            ),
            {"tenant_id": str(tenant_id)},
        ).scalar_one_or_none()
        if active is not True:
            raise InactiveTenantError("Workspace is inactive or missing")


@contextmanager
def tenant_connection(
    tenant_id, isolation_level: str | None = None, *, require_active: bool = True
):
    """Open one transaction with the RLS workspace setting installed."""
    conn = get_app_engine().connect()
    try:
        if isolation_level is not None:
            conn = conn.execution_options(isolation_level=isolation_level)
        with conn.begin():
            conn.execute(
                text("SELECT pg_advisory_xact_lock_shared(:lock_id)"),
                {"lock_id": _lock_id(tenant_id)},
            )
            _set_tenant_context(conn, tenant_id, require_active=require_active)
            yield conn
    finally:
        conn.close()


@contextmanager
def tenant_graph_write_lock(tenant_id):
    """Serialize Graphiti mutations for one workspace across processes."""
    conn = get_app_engine().connect()
    lock_id = _lock_id(tenant_id, "graph-write")
    try:
        conn.execute(text("SELECT pg_advisory_lock(:lock_id)"), {"lock_id": lock_id})
        conn.commit()
        try:
            yield
        finally:
            try:
                conn.execute(
                    text("SELECT pg_advisory_unlock(:lock_id)"),
                    {"lock_id": lock_id},
                )
                conn.commit()
            except Exception:
                conn.invalidate()
                raise
    finally:
        conn.close()


def resolve_api_key(api_key: str) -> SimpleNamespace | None:
    """Resolve a bearer key by SHA-256 digest without exposing stored hashes."""
    if not isinstance(api_key, str):
        return None
    encoded = api_key.encode("utf-8")
    if not encoded or len(encoded) > 1024:
        return None
    with get_app_engine().begin() as conn:
        row = conn.execute(
            text(
                "SELECT tenant_id, key_id, key_kind, key_label, scopes, "
                "allowed_subject_scopes, timezone, locale, expires_at "
                "FROM private.resolve_api_key_hash(:key_hash)"
            ),
            {"key_hash": hashlib.sha256(encoded).digest()},
        ).one_or_none()
    if row is None:
        return None
    return SimpleNamespace(
        tenant_id=row.tenant_id,
        key_id=row.key_id,
        key_kind=row.key_kind,
        key_label=row.key_label,
        scopes=tuple(row.scopes or ()),
        allowed_subject_scopes=row.allowed_subject_scopes,
        timezone=row.timezone,
        locale=row.locale,
        expires_at=row.expires_at,
    )


def get_tenant_settings(tenant_id) -> SimpleNamespace:
    with tenant_connection(tenant_id) as conn:
        row = conn.execute(
            text(
                "SELECT timezone, locale, active FROM tenant_settings "
                "WHERE tenant_id=:tenant_id"
            ),
            {"tenant_id": str(tenant_id)},
        ).one()
    return SimpleNamespace(timezone=row.timezone, locale=row.locale, active=row.active)
