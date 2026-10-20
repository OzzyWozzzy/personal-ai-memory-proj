"""POST /v1/context — the two-route retrieval + budgeted assembly endpoint.

Auth and tenant context are applied by the app factory. Route decisions and
policy filtering are logged to ``audit_log`` by the retrieval engine.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, model_validator

from app import db
from app.engine.retrieval import DEFAULT_BUDGET, assemble_context

router = APIRouter(prefix="/v1", tags=["context"])

class RefInterval(BaseModel):
    """A reference interval ``[start, end)``; ``end=null`` means +∞ (open)."""
    start: datetime
    end: Optional[datetime] = None


class ContextRequest(BaseModel):
    query: str
    subject_scope: Optional[UUID] = None
    token_budget: int = Field(default=DEFAULT_BUDGET, ge=0)
    # tier_floor is DEAD (authority tiers removed, spec v3.2) — accepted and
    # ignored for back-compat so old clients don't 422.
    tier_floor: Optional[int] = None
    as_of: Optional[datetime] = None
    as_believed_at: Optional[datetime] = None
    # Retained in the schema only to return a clear migration error. Graphiti's
    # native search exposes date filters, not Allen's 13 interval operations.
    interval_op: Optional[str] = None
    ref_interval: Optional[RefInterval] = None
    include_historical: bool = False
    include_disputed: bool = False
    include: Optional[Literal["facts", "knowledge", "narrative", "merged"]] = None
    format: Literal["block", "json"] = "block"

    @model_validator(mode="after")
    def _check_interval(self):
        if self.interval_op is not None or self.ref_interval is not None:
            raise ValueError(
                "interval_op/ref_interval were retired in Graphiti-native mode; "
                "use as_of and include_historical"
            )
        return self


@router.post("/context")
def post_context(req: ContextRequest, request: Request) -> dict:
    ctx = request.state.tenant_ctx
    ref = None
    if req.ref_interval is not None:
        ref = {"start": req.ref_interval.start, "end": req.ref_interval.end}
    with db.tenant_connection(ctx.tenant_id) as conn:
        return assemble_context(
            conn, ctx.tenant_id,
            query=req.query,
            subject_scope=req.subject_scope,
            token_budget=req.token_budget,
            as_of=req.as_of,
            as_believed_at=req.as_believed_at,
            interval_op=req.interval_op,
            ref_interval=ref,
            include_historical=req.include_historical,
            include_disputed=req.include_disputed,
            include=req.include,
            fmt=req.format,
            # ABAC (§8.1): the key's subject-scope allowlist (None = unrestricted).
            allowed_subject_scopes=getattr(ctx, "allowed_subject_scopes", None),
        )
