"""Credential lifecycle helpers for the local graph inspector."""
from __future__ import annotations

import hashlib

from fastapi import Request, Response
from sqlalchemy import text

from app import db
from app.security import DEV_PREVIEW_COOKIE


def revoke_preview_key(raw_key: str | None) -> None:
    """Revoke the presented preview credential (and any stale preview keys).

    The workspace is persistent, so rotating the browser credential does not
    revoke user-created agent keys.
    """
    if not raw_key:
        return
    digest = hashlib.sha256(raw_key.encode("utf-8")).digest()
    with db.get_admin_engine().begin() as conn:
        # Locking the presented row serializes with preview-authorized agent-key
        # creation and gives the UPDATE a fresh READ COMMITTED snapshot.
        tenant_id = conn.execute(
            text(
                "SELECT tenant_id FROM api_keys"
                " WHERE key_hash=:key_hash AND name='preview'"
                " FOR UPDATE"
            ),
            {"key_hash": digest},
        ).scalar_one_or_none()
        if tenant_id is None:
            return
        conn.execute(
            text(
                "UPDATE api_keys SET active=false"
                " WHERE tenant_id=:tenant_id AND name = 'preview'"
            ),
            {"tenant_id": str(tenant_id)},
        )


def clear_preview_session(request: Request, response: Response) -> None:
    """Revoke and clear the browser's local credential."""
    revoke_preview_key(request.cookies.get(DEV_PREVIEW_COOKIE))
    response.delete_cookie(
        DEV_PREVIEW_COOKIE,
        path="/",
        httponly=True,
        samesite="strict",
    )
