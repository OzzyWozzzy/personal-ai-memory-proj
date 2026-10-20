"""Persistent local workspace for the bundled graph inspector.

The demo is ONE stable tenant that survives across clicks, restarts, and
sessions — whatever the user saves in the demo stays saved. "Continue with
demo" re-attaches to that workspace (creating it empty on first use; nothing
is seeded) and rotates the preview credential. The tenant is pinned by
``MEMORY_PREVIEW_TENANT_ID`` when set (e.g. to share one memory with an MCP
client's key), else derived from a stable local namespace.

The browser receives workspace metadata but never the preview credential. A
random machine key is stored hash-only and transported solely in an HttpOnly,
SameSite=Strict cookie. The key expires after eight hours. It includes the
admin capability needed by the local governance view.
"""
from __future__ import annotations

import hashlib
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request, Response
from sqlalchemy import text

from app import db
from app.local_preview import clear_preview_session, revoke_preview_key
from app.security import DEV_PREVIEW_COOKIE, local_preview_enabled


public_router = APIRouter(prefix="/v1/auth/preview", tags=["local-workspace"])
_LIFETIME = timedelta(hours=8)

#: Stable fallback so the demo workspace persists without any configuration.
_DEFAULT_PREVIEW_TENANT = uuid.uuid5(
    uuid.NAMESPACE_URL, "temporal-memory-graph:local-preview"
)


def _require_enabled() -> None:
    if not local_preview_enabled():
        raise HTTPException(404, "Not found")


def _require_same_origin(request: Request) -> None:
    """Prevent third-party pages from creating local tenants through a browser."""
    if request.headers.get("sec-fetch-site", "").lower() == "cross-site":
        raise HTTPException(403, "Cross-site preview request rejected")
    origin = request.headers.get("origin")
    if not origin:
        raise HTTPException(403, "Preview requires a same-origin browser request")
    try:
        parsed = urlparse(origin)
        origin_host = parsed.netloc.casefold()
    except ValueError:
        raise HTTPException(403, "Cross-site preview request rejected")
    if parsed.scheme not in {"http", "https"} or origin_host != request.headers.get(
        "host", ""
    ).casefold():
        raise HTTPException(403, "Cross-site preview request rejected")


def _preview_tenant_id() -> uuid.UUID:
    raw = (os.environ.get("MEMORY_PREVIEW_TENANT_ID") or "").strip()
    if not raw:
        return _DEFAULT_PREVIEW_TENANT
    try:
        return uuid.UUID(raw)
    except ValueError:
        raise HTTPException(
            503, "Local preview is misconfigured (MEMORY_PREVIEW_TENANT_ID)"
        )


def _live_counts(tenant_id: uuid.UUID) -> dict:
    """Current local workspace counts."""
    from app.engine import graph, views

    with db.tenant_connection(tenant_id) as conn:
        episodes = conn.execute(
            text("SELECT count(*) FROM episodes WHERE tenant_id=:t"),
            {"t": str(tenant_id)},
        ).scalar_one()
    facts = (
        views.fact_counts(str(tenant_id))[0]
        if graph.graph_exists(str(tenant_id))
        else 0
    )
    return {"episodes": int(episodes), "facts": int(facts)}


@public_router.get("")
def capability(response: Response) -> dict:
    _require_enabled()
    response.headers["Cache-Control"] = "no-store"
    return {"enabled": True}


@public_router.post("", status_code=201)
def start_preview(request: Request, response: Response) -> dict:
    _require_enabled()
    _require_same_origin(request)
    # Re-clicking rotates and revokes the former preview credential; the
    # WORKSPACE persists — only the key changes.
    revoke_preview_key(request.cookies.get(DEV_PREVIEW_COOKIE))
    tenant_id = _preview_tenant_id()
    raw_key = "mem_preview_" + secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + _LIFETIME
    with db.get_admin_engine().begin() as conn:
        conn.execute(
            text(
                "INSERT INTO api_keys"
                " (key_hash, key_prefix, tenant_id, name, label, scopes, expires_at)"
                " VALUES (:key_hash, :prefix, :tenant_id, 'preview',"
                " 'Local workspace',"
                " ARRAY['memory:read','memory:write','workspace:admin']::text[],"
                " :expires_at)"
            ),
            {
                "key_hash": hashlib.sha256(raw_key.encode("utf-8")).digest(),
                "prefix": raw_key[:16],
                "tenant_id": str(tenant_id),
                "expires_at": expires_at,
            },
        )
        # First click creates the workspace; every later click re-attaches.
        conn.execute(
            text(
                "INSERT INTO tenant_settings (tenant_id, timezone, locale)"
                " VALUES (:tenant_id, 'UTC', 'en')"
                " ON CONFLICT (tenant_id) DO NOTHING"
            ),
            {"tenant_id": str(tenant_id)},
        )
    try:
        counts = _live_counts(tenant_id)
    except Exception as exc:
        revoke_preview_key(raw_key)
        raise HTTPException(503, "Local preview could not be prepared") from exc

    # The credential itself never appears in the JSON body.
    response.set_cookie(
        DEV_PREVIEW_COOKIE,
        raw_key,
        max_age=int(_LIFETIME.total_seconds()),
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="strict",
        path="/",
    )
    response.headers["Cache-Control"] = "no-store"
    return {
        "preview": True,
        "workspace": {
            "id": str(tenant_id),
            "name": "Local Memory",
            "role": "local",
            "counts": counts,
        },
    }


@public_router.delete("")
def stop_preview(request: Request, response: Response) -> dict:
    _require_enabled()
    _require_same_origin(request)
    clear_preview_session(request, response)
    response.headers["Cache-Control"] = "no-store"
    response.headers["Clear-Site-Data"] = '"cache", "cookies", "storage"'
    return {"preview": False}
