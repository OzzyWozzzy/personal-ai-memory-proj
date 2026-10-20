"""Readable chronological history for one graph entity."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request

from app.engine import views
from app.humanize import humanize_type

router = APIRouter()
_STORY_EDGE_CAP = 500


@router.get("/v1/subjects/{subject_id}/story")
def subject_story(subject_id: UUID, request: Request) -> dict:
    tenant_id = request.state.tenant_ctx.tenant_id
    entity = views.fetch_entity(tenant_id, subject_id)
    if entity is None:
        raise HTTPException(404, "Unknown graph entity")
    name, labels = entity
    edges = views.fetch_entity_edges(tenant_id, subject_id, _STORY_EDGE_CAP)
    entries = views.story_entries(edges, now=datetime.now(tz=timezone.utc))
    entries.sort(key=lambda item: (item["_at"], item["kind"], item["text"]))
    for entry in entries:
        entry.pop("_at", None)
    return {
        "subject_id": str(subject_id),
        "name": name,
        "kind": humanize_type(views.entity_type_of(labels)),
        "story": entries,
    }
