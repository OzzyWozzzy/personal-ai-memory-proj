"""Local-first FastAPI application for the temporal memory engine.

The HTTP surface is intentionally explicit. This is a reference implementation,
not a hosted multi-user product: callers use either a local preview cookie (the
bundled UI) or a hash-only API key (scripts and MCP).
"""
from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app import db
from app.security import (
    DEV_PREVIEW_COOKIE,
    SecurityConfig,
    install_security_middleware,
    local_preview_enabled,
)


@asynccontextmanager
async def _application_lifespan(_app: FastAPI):
    try:
        yield
    finally:
        from app.engine import graph

        graph.close_all()


def require_api_key(
    request: Request, x_api_key: str | None = Header(default=None)
) -> SimpleNamespace:
    """Resolve a hash-only API key into the repository's tenant context."""
    if not x_api_key:
        raise HTTPException(401, "Missing X-API-Key header")
    resolved = db.resolve_api_key(x_api_key)
    if resolved is None:
        raise HTTPException(401, "Unknown API key")
    tenant_ctx = SimpleNamespace(
        tenant_id=resolved.tenant_id,
        timezone=resolved.timezone,
        locale=resolved.locale,
        api_key=f"key:{resolved.key_id}",
        auth_kind="api_key",
        role="agent",
        user_id=None,
        key_kind=getattr(resolved, "key_kind", ""),
        key_label=getattr(resolved, "key_label", ""),
        key_id=getattr(resolved, "key_id", None),
        scopes=frozenset(getattr(resolved, "scopes", ())),
        allowed_subject_scopes=getattr(resolved, "allowed_subject_scopes", None),
    )
    request.state.tenant_ctx = tenant_ctx
    return tenant_ctx


def _require_scope(request: Request, tenant_ctx: SimpleNamespace) -> None:
    if request.url.path.startswith(("/v1/admin", "/v1/agent_keys")):
        required = "workspace:admin"
    elif request.method in {"GET", "HEAD", "OPTIONS"}:
        required = "memory:read"
    else:
        required = "memory:write"
    if required not in tenant_ctx.scopes:
        raise HTTPException(403, "API key lacks the required capability")


def require_principal(
    request: Request,
    x_api_key: str | None = Header(default=None),
) -> SimpleNamespace:
    """Accept one machine key or the local UI's HttpOnly preview key."""
    preview_key = (
        request.cookies.get(DEV_PREVIEW_COOKIE) if local_preview_enabled() else None
    )
    if preview_key and x_api_key:
        raise HTTPException(400, "Conflicting authentication credentials")

    if preview_key:
        tenant_ctx = require_api_key(request, preview_key)
        if tenant_ctx.key_kind != "preview":
            raise HTTPException(401, "Local session is invalid")
        tenant_ctx.auth_kind = "local_preview"
        tenant_ctx.role = "local"
        _require_scope(request, tenant_ctx)
        return tenant_ctx

    if x_api_key:
        tenant_ctx = require_api_key(request, x_api_key)
        _require_scope(request, tenant_ctx)
        return tenant_ctx

    raise HTTPException(401, "Start the local UI session or provide X-API-Key")


def create_app() -> FastAPI:
    security = SecurityConfig.from_env()
    app = FastAPI(
        title="Temporal Memory Graph",
        description="Local-first source-grounded temporal memory.",
        version="0.1.0",
        lifespan=_application_lifespan,
    )
    app.state.security_config = security
    install_security_middleware(app, security)

    @app.exception_handler(db.InactiveTenantError)
    async def inactive_workspace_handler(_request: Request, _exc: Exception):
        return JSONResponse(status_code=404, content={"detail": "Workspace not found"})

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    readiness_lock = threading.Lock()
    readiness_cache = {"checked_at": 0.0, "ready": False}

    @app.get("/readyz")
    def readyz():
        now = time.monotonic()
        with readiness_lock:
            max_age = 10.0 if readiness_cache["ready"] else 2.0
            if now - readiness_cache["checked_at"] >= max_age:
                try:
                    with db.get_app_engine().connect() as conn:
                        ready = conn.execute(text("SELECT 1")).scalar_one() == 1
                    if ready:
                        from app.engine import graph

                        graph.health_check()
                except Exception:
                    ready = False
                readiness_cache.update(checked_at=now, ready=ready)
            ready = bool(readiness_cache["ready"])
        return JSONResponse(
            status_code=200 if ready else 503,
            content={"status": "ready" if ready else "not ready"},
            headers={"Cache-Control": "no-store"},
        )

    # One write path, two retrieval paths, temporal inspection, and governance.
    # Nothing is auto-discovered.
    from app.api import (
        admin,
        agent_access,
        context,
        governance,
        graph,
        remember,
        search,
        stats,
        story,
        subjects,
    )

    for module in (
        remember,
        context,
        search,
        graph,
        subjects,
        story,
        stats,
        governance,
        admin,
        agent_access,
    ):
        app.include_router(module.router, dependencies=[Depends(require_principal)])

    from app.api.ui import ui_router

    app.include_router(ui_router)
    if local_preview_enabled():
        from app.api.local_preview import public_router

        app.include_router(public_router)
    return app
