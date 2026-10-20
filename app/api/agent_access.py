"""Tenant-scoped, hash-only API keys for MCP servers, bots, and scripts.

``POST /v1/agent_keys`` returns a high-entropy bearer key exactly once.  Only
its SHA-256 digest is stored.  List/revoke responses use an opaque ``key_id``
and a deliberately partial, non-secret ``key_prefix``; raw credentials and
stored hashes never leave the creation response or database resolver boundary.

The table has FORCE RLS (migration 0011).  Every operation below runs through
``tenant_connection`` and is additionally limited to ``name='agent'``.  Rows
are revoked rather than deleted so lifecycle metadata remains auditable.
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text

from app.audit import write_audit
from app.db import tenant_connection

router = APIRouter(prefix="/v1/agent_keys", tags=["agent_access"])

_TOKEN_PREFIX = "mem_agent_"
_STORED_PREFIX_CHARS = len(_TOKEN_PREFIX) + 8
_AGENT_SCOPES = ("memory:read", "memory:write")


def _preview(key_prefix: str) -> str:
    return key_prefix + "…"


class AgentKeyIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(default="", max_length=100)
    expires_in_days: int | None = Field(default=None, ge=1, le=365)


class RevokeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key_id: UUID | None = None
    key_prefix: str | None = Field(default=None, max_length=33)


@router.post("", status_code=201)
def create_agent_key(body: AgentKeyIn, request: Request) -> dict:
    ctx = request.state.tenant_ctx
    # token_urlsafe(32) carries 256 random bits; unlike the old 64-bit token it
    # remains unguessable even after a non-secret display prefix is disclosed.
    api_key = _TOKEN_PREFIX + secrets.token_urlsafe(32)
    key_prefix = api_key[:_STORED_PREFIX_CHARS]
    key_hash = hashlib.sha256(api_key.encode("utf-8")).digest()
    label = body.label.strip()
    now = datetime.now(timezone.utc)
    expires_at = (
        now + timedelta(days=body.expires_in_days)
        if body.expires_in_days is not None
        else None
    )
    with tenant_connection(ctx.tenant_id) as conn:
        if ctx.auth_kind == "local_preview":
            preview_expires_at = conn.execute(
                text(
                    "SELECT expires_at FROM api_keys"
                    " WHERE tenant_id=:tenant_id AND key_id=:key_id"
                    " AND name='preview' AND active"
                    " FOR UPDATE"
                ),
                {
                    "tenant_id": str(ctx.tenant_id),
                    "key_id": str(ctx.key_id),
                },
            ).scalar_one_or_none()
            if preview_expires_at is None or preview_expires_at <= now:
                raise HTTPException(401, "Preview session has expired.")
            # A copied demo key must never outlive the isolated preview that
            # authorized it, even if the browser closes without calling Exit.
            if expires_at is None or preview_expires_at < expires_at:
                expires_at = preview_expires_at
        row = conn.execute(
            text(
                """
                INSERT INTO api_keys
                  (key_hash, key_prefix, tenant_id, name, label, scopes,
                   expires_at)
                VALUES
                  (:key_hash, :key_prefix, :tenant_id, 'agent', :label,
                   ARRAY['memory:read', 'memory:write']::text[], :expires_at)
                RETURNING key_id
                """
            ),
            {
                "key_hash": key_hash,
                "key_prefix": key_prefix,
                "tenant_id": str(ctx.tenant_id),
                "label": label,
                "expires_at": expires_at,
            },
        ).one()
        write_audit(
            conn,
            ctx.tenant_id,
            "agent_keys",
            "agent_key.created",
            {
                "key_id": str(row.key_id),
                "label": label,
                "key_preview": _preview(key_prefix),
                "scopes": list(_AGENT_SCOPES),
                "expires_at": expires_at.isoformat() if expires_at else None,
            },
        )
    return {
        "key_id": str(row.key_id),
        "api_key": api_key,
        "label": label,
        "key_prefix": key_prefix,
        "key_preview": _preview(key_prefix),
        "scopes": list(_AGENT_SCOPES),
        "expires_at": expires_at.isoformat() if expires_at else None,
        "note": "Copy this key now — it won't be shown again.",
    }


@router.get("")
def list_agent_keys(request: Request) -> dict:
    ctx = request.state.tenant_ctx
    with tenant_connection(ctx.tenant_id) as conn:
        rows = conn.execute(
            text(
                """
                SELECT key_id, key_prefix, label, active, scopes, created_at,
                       expires_at, last_used_at
                  FROM api_keys
                 WHERE tenant_id = :tenant_id AND name = 'agent'
                 ORDER BY created_at, key_id
                """
            ),
            {"tenant_id": str(ctx.tenant_id)},
        ).all()
    return {
        "keys": [
            {
                "key_id": str(row.key_id),
                "label": row.label,
                "key_prefix": row.key_prefix,
                "key_preview": _preview(row.key_prefix),
                "active": row.active,
                "scopes": list(row.scopes or ()),
                "created_at": row.created_at.isoformat(),
                "expires_at": (
                    row.expires_at.isoformat() if row.expires_at else None
                ),
                "last_used_at": (
                    row.last_used_at.isoformat() if row.last_used_at else None
                ),
            }
            for row in rows
        ]
    }


@router.post("/revoke")
def revoke_agent_key(body: RevokeIn, request: Request) -> dict:
    ctx = request.state.tenant_ctx
    prefix = (body.key_prefix or "").strip().rstrip("…").strip()
    if (body.key_id is None) == (not prefix):
        raise HTTPException(
            422, "Provide exactly one key_id or the key_prefix shown in the list."
        )

    params = {
        "tenant_id": str(ctx.tenant_id),
        "selector_id": str(body.key_id) if body.key_id is not None else None,
        "selector_prefix": None,
    }
    if body.key_id is None:
        # Prefixes returned by the list are exact stored identifiers.  We do
        # not accept raw credentials or search against raw/hash material.
        if len(prefix) < 3:
            raise HTTPException(422, "The key prefix is too short.")
        params["selector_prefix"] = prefix

    with tenant_connection(ctx.tenant_id) as conn:
        rows = conn.execute(
            text(
                """
                SELECT key_id, key_prefix, label
                  FROM api_keys
                 WHERE tenant_id = :tenant_id
                   AND name = 'agent'
                   AND active
                   AND (
                     (CAST(:selector_id AS uuid) IS NOT NULL
                      AND key_id = CAST(:selector_id AS uuid))
                     OR
                     (CAST(:selector_prefix AS text) IS NOT NULL
                      AND key_prefix = CAST(:selector_prefix AS text))
                   )
                """
            ),
            params,
        ).all()
        if not rows:
            raise HTTPException(404, "Active agent key not found.")
        if len(rows) > 1:
            raise HTTPException(
                409, "That prefix matches more than one key; revoke by key_id."
            )
        key = rows[0]
        conn.execute(
            text(
                """
                UPDATE api_keys SET active = false
                 WHERE tenant_id = :tenant_id AND key_id = :key_id
                """
            ),
            {"tenant_id": str(ctx.tenant_id), "key_id": str(key.key_id)},
        )
        write_audit(
            conn,
            ctx.tenant_id,
            "agent_keys",
            "agent_key.revoked",
            {
                "key_id": str(key.key_id),
                "label": key.label,
                "key_preview": _preview(key.key_prefix),
            },
        )
    return {
        "revoked": True,
        "key_id": str(key.key_id),
        "label": key.label,
        "key_prefix": key.key_prefix,
        "key_preview": _preview(key.key_prefix),
        "note": "This key stops working immediately.",
    }
