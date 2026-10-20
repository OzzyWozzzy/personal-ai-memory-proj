"""Immutable Postgres episode ledger for the ingestion path.

``ingest_episode`` computes a canonical ``content_hash``, dedupes per workspace
on it, and inserts the immutable source episode. Graph extraction is owned by
the engine write path; there is no parallel background embedding queue.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from uuid import UUID
from zoneinfo import ZoneInfo

from sqlalchemy import text

_UTC = timezone.utc


def canonical_hash(content: str | None, content_json) -> str:
    """sha256 over a canonical serialization: raw ``content`` when present, else
    the sorted-key JSON of ``content_json``. Equal payloads hash identically."""
    if content is not None:
        payload = content
    else:
        payload = json.dumps(content_json, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_occurred_at(raw: str | None, tz: str = "UTC") -> datetime:
    """ISO string -> UTC-aware datetime (naive read in tenant tz). None -> now()."""
    if raw is None:
        return datetime.now(tz=_UTC)
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(tz))
    return dt.astimezone(_UTC)


def ingest_episode(conn, tenant_id, *, source: str, occurred_at: datetime,
                   content: str | None = None, content_json=None,
                   source_ref: str | None = None, subject_scope=None,
                   meta: dict | None = None,
                   content_hash: str | None = None) -> tuple[UUID, bool]:
    """Insert-or-dedupe one episode. Returns ``(episode_id, deduped)``.

    Idempotent on ``(tenant_id, content_hash)`` via ``ON CONFLICT DO NOTHING``;
    a losing concurrent insert falls back to selecting the winner's id. The
    The database uniqueness constraint also makes concurrent retries safe.
    """
    chash = content_hash or canonical_hash(content, content_json)
    row = conn.execute(
        text(
            "INSERT INTO episodes (tenant_id, subject_scope, source, source_ref,"
            " occurred_at, content, content_json, content_hash, meta)"
            " VALUES (:tid, :scope, :source, :sref, :occ, :content,"
            " cast(:cjson as jsonb), :chash, cast(:meta as jsonb))"
            " ON CONFLICT (tenant_id, content_hash) DO NOTHING"
            " RETURNING episode_id"
        ),
        {"tid": str(tenant_id),
         "scope": str(subject_scope) if subject_scope else None,
         "source": source, "sref": source_ref, "occ": occurred_at,
         "content": content,
         "cjson": json.dumps(content_json) if content_json is not None else None,
         "chash": chash, "meta": json.dumps(meta or {})},
    ).first()
    if row is not None:
        return row.episode_id, False
    existing = conn.execute(
        text("SELECT episode_id FROM episodes"
             " WHERE tenant_id=:tid AND content_hash=:chash"),
        {"tid": str(tenant_id), "chash": chash},
    ).scalar_one()
    return existing, True


def json_safe(value):
    """Recursively coerce a row/mapping to JSON-serializable primitives:
    UUID -> str, datetime -> ISO 8601, dict/list recursed, everything else
    passed through (psycopg already decodes jsonb and tsvector to str)."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value
