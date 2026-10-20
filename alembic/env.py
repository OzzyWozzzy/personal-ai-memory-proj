"""Alembic environment.

The migration URL comes from ``app.db.get_admin_url()`` (NOT a hardcoded ini
value), so the same migrations apply to the embedded pgserver dev DB or to
docker-compose depending only on environment variables.
"""
from __future__ import annotations

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine

# Make the repo root importable regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import get_admin_url  # noqa: E402

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# No SQLAlchemy metadata: the schema is raw DDL for provability, so autogenerate
# is not used.
target_metadata = None


def run_migrations_offline() -> None:
    context.configure(
        url=get_admin_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # Data-contract migrations bind encryption material as parameters. Even if
    # an operator temporarily enables SQL logging, values must never appear in
    # DBAPI/SQLAlchemy exceptions or migration output.
    engine = create_engine(
        get_admin_url(), pool_pre_ping=True, hide_parameters=True)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
