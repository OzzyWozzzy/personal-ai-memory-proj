"""Embedded dev/test PostgreSQL bootstrap via pgserver.

Boots a real PostgreSQL 16 under ``<repo>/.devdb/pgdata``
and returns SQLAlchemy connection URLs. This is the zero-Docker local fallback
used by ``app.db`` when ``POSTGRES_ADMIN_URL`` / ``POSTGRES_APP_URL`` are unset.
Tests set ``MEMORY_DEVDB_DIR`` to an isolated temporary root so fixture rows can
never reach the persistent local application database.

pgserver runs ``initdb`` with trust auth on a unix socket, so any role — including
the non-OS ``app_user`` — connects without pg_hba edits. The socket directory is
dynamic (macOS path-length workaround), so it is always read at runtime from the
running postmaster, never hardcoded.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import psycopg
from sqlalchemy.engine import URL

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PGDATA = _REPO_ROOT / ".devdb" / "pgdata"

_DB_NAME = "memory"
_APP_ROLE = "app_user"

_lock = threading.Lock()
# Module-level singletons: keep the server handle alive (cleanup_mode='stop'
# stops the server when the last handle is dropped) and memoize the URLs.
_server = None
_urls: tuple[str, str] | None = None


def _sa_url(user: str, password: str | None, socket_dir: str) -> str:
    """Build a SQLAlchemy psycopg URL that connects over the unix socket."""
    return URL.create(
        "postgresql+psycopg",
        username=user,
        password=password,
        database=_DB_NAME,
        query={"host": socket_dir},
    ).render_as_string(hide_password=False)


def ensure_devdb() -> tuple[str, str]:
    """Start the embedded server (idempotent) and return ``(admin_url, app_url)``.

    admin_url connects as ``postgres`` (superuser, bypasses RLS) to run migrations
    and auth-root lookups; app_url connects as the non-superuser ``app_user`` that
    RLS is enforced against. Idempotent across calls (memoized) and across process
    runs (existence-guarded CREATE DATABASE / CREATE ROLE, datadir persists).
    """
    global _server, _urls
    with _lock:
        if _urls is not None:
            return _urls

        import pgserver

        override = os.environ.get("MEMORY_DEVDB_DIR", "").strip()
        pgdata = Path(override).expanduser().resolve() / "pgdata" \
            if override else _DEFAULT_PGDATA
        pgdata.mkdir(parents=True, exist_ok=True)
        _server = pgserver.get_server(pgdata, cleanup_mode="stop")
        socket_dir = str(_server.get_postmaster_info().socket_dir)

        # Bootstrap the database and app role against the default 'postgres' db.
        # CREATE DATABASE / CREATE ROLE cannot be parameterized and must run
        # outside a transaction block -> autocommit.
        bootstrap = f"postgresql://postgres@/postgres?host={socket_dir}"
        with psycopg.connect(bootstrap, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (_DB_NAME,))
            if cur.fetchone() is None:
                cur.execute(f'CREATE DATABASE "{_DB_NAME}"')
            cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (_APP_ROLE,))
            if cur.fetchone() is None:
                cur.execute(
                    "CREATE ROLE app_user LOGIN NOSUPERUSER NOBYPASSRLS NOINHERIT"
                )

        _urls = (
            _sa_url("postgres", None, socket_dir),
            # The embedded server is bound to a repository-local Unix socket
            # with trust auth. Supplying even a development password would be
            # misleading and creates an unnecessary scanner finding.
            _sa_url(_APP_ROLE, None, socket_dir),
        )
        return _urls
