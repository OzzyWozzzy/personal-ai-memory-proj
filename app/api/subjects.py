"""GET /v1/subjects/{subject_id}/history — a subject's edge history.

The raw record behind ``/story``: the same edge set (all of the subject's edges,
both directions, from the graph), rendered as JSON-safe edge views ordered oldest
first. ``relation`` is now OPTIONAL — when given it filters to that relation
(name-normalized so ``works_at`` / ``works at`` match); when omitted, every
relation is returned.
"""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Request

from app.engine import views

router = APIRouter()

#: Cap on edges pulled for one subject (bounds a hub entity's fan-out).
_HISTORY_EDGE_CAP = 500


@router.get("/v1/subjects/{subject_id}/history")
def subject_history(subject_id: UUID, request: Request,
                    relation: str | None = None) -> dict:
    ctx = request.state.tenant_ctx
    edges = views.fetch_entity_edges(ctx.tenant_id, subject_id, _HISTORY_EDGE_CAP)
    edges = [e for e in edges if views.matches_relation(e, relation)]
    return {
        "subject_id": str(subject_id),
        "relation": relation,
        "history": views.history_entries(edges),
    }
