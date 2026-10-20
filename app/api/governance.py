"""Governed eviction, legal hold, and retention control (design plan §8.4).

Every endpoint routes deletion through :mod:`app.eviction` (the sole DELETE
path). Auth and tenant context are supplied by the app factory.

* ``POST /v1/admin/episodes/{id}/evict?dry_run=`` — governed lineage eviction.
  ``dry_run=true`` returns the plan without mutating. A legal-hold episode →
  ``409``.
* ``POST /v1/admin/episodes/{id}/legal_hold`` ``{held}`` — set/clear the hold.
* ``POST /v1/admin/clear?dry_run=`` — clear every evictable episode; held
  episodes are skipped and reported.
* ``POST /v1/admin/retention/sweep?dry_run=`` — evict past-retention episodes
  (defaults to ``dry_run=true``, a planning call).
* ``GET  /v1/admin/retention`` — the tenant's current ``source -> days`` policy.
* ``PUT  /v1/admin/retention`` ``{source: days, ...}`` — replace the policy.
"""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import text

from app import db, eviction

router = APIRouter(prefix="/v1/admin", tags=["governance"])


class LegalHoldIn(BaseModel):
    held: bool


class RetentionIn(BaseModel):
    # source -> max_age_days (integers). {} keeps everything forever.
    retention: dict[str, int]


@router.post("/episodes/{episode_id}/evict")
def evict_episode(episode_id: UUID, request: Request,
                  dry_run: bool = Query(default=False)) -> dict:
    ctx = request.state.tenant_ctx
    try:
        with db.tenant_connection(ctx.tenant_id) as conn:
            plan = eviction.evict_episode(
                conn, ctx.tenant_id, episode_id,
                actor=f"admin:{ctx.api_key}", dry_run=dry_run)
    except eviction.EvictionBlocked as exc:
        raise HTTPException(409, str(exc))
    except eviction.EvictionError as exc:
        raise HTTPException(404, str(exc))
    return {"episode_id": str(episode_id), "dry_run": dry_run, "plan": plan}


@router.post("/clear")
def clear_workspace(request: Request, dry_run: bool = Query(default=False)) -> dict:
    """Clear the workspace through the governed eviction path.
    Legal-held episodes (and their lineage) are skipped, not an error — the
    response reports them in ``held_episodes``."""
    ctx = request.state.tenant_ctx
    with db.tenant_connection(ctx.tenant_id) as conn:
        totals = eviction.clear_tenant(
            conn, ctx.tenant_id, actor=f"admin:{ctx.api_key}", dry_run=dry_run)
    return {"dry_run": dry_run, **totals}


@router.post("/episodes/{episode_id}/legal_hold")
def set_legal_hold(episode_id: UUID, body: LegalHoldIn, request: Request) -> dict:
    ctx = request.state.tenant_ctx
    try:
        with db.tenant_connection(ctx.tenant_id) as conn:
            eviction.set_legal_hold(conn, ctx.tenant_id, episode_id, body.held,
                                    actor=f"admin:{ctx.api_key}")
    except eviction.EvictionError as exc:
        raise HTTPException(404, str(exc))
    return {"episode_id": str(episode_id), "legal_hold": body.held}


@router.post("/retention/sweep")
def retention_sweep(request: Request, dry_run: bool = Query(default=True)) -> dict:
    ctx = request.state.tenant_ctx
    with db.tenant_connection(ctx.tenant_id) as conn:
        return eviction.retention_sweep(
            conn, ctx.tenant_id, actor=f"admin:{ctx.api_key}", dry_run=dry_run)


@router.get("/retention")
def get_retention(request: Request) -> dict:
    ctx = request.state.tenant_ctx
    with db.tenant_connection(ctx.tenant_id) as conn:
        row = conn.execute(
            text("SELECT retention FROM tenant_policies WHERE tenant_id=:t"),
            {"t": str(ctx.tenant_id)}).first()
    return {"retention": (row.retention if row else None) or {}}


@router.put("/retention")
def put_retention(body: RetentionIn, request: Request) -> dict:
    import json

    ctx = request.state.tenant_ctx
    with db.tenant_connection(ctx.tenant_id) as conn:
        conn.execute(
            text("INSERT INTO tenant_policies (tenant_id, retention)"
                 " VALUES (:t, cast(:r as jsonb))"
                 " ON CONFLICT (tenant_id) DO UPDATE SET retention=excluded.retention"),
            {"t": str(ctx.tenant_id), "r": json.dumps(body.retention)})
    return {"retention": body.retention}
