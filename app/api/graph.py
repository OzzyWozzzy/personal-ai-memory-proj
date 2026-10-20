"""Temporal graph projection used by the local visual inspector.

Entity edges come directly from Graphiti/FalkorDB. Optional episode nodes come
from the immutable Postgres ledger and make source provenance inspectable
without pretending raw records are extracted facts.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime

from fastapi import APIRouter, Query, Request
from sqlalchemy import text

from app import db
from app.engine import views

router = APIRouter(prefix="/v1", tags=["graph"])
_ENTRY_GRAPH_LIMIT = 1000


def _iso(value) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None


@router.get("/graph")
def get_graph(
    request: Request,
    as_of: datetime | None = Query(default=None),
    include_historical: bool = Query(default=False),
    include_episodes: bool = Query(default=False),
    limit: int = Query(default=500, ge=1, le=5000),
) -> dict:
    ctx = request.state.tenant_ctx
    scopes = getattr(ctx, "allowed_subject_scopes", None)
    scope_list = None if scopes is None else [str(item) for item in scopes]
    with db.tenant_connection(ctx.tenant_id) as conn:
        at = as_of or conn.execute(text("SELECT now()")).scalar_one()
        edges, nodes, counts = _build_graph(
            conn,
            ctx.tenant_id,
            at,
            include_historical=include_historical,
            include_episodes=include_episodes,
            scope_list=scope_list,
            limit=limit,
        )
    return {
        "as_of": _iso(as_of),
        "at": _iso(at),
        "include_historical": include_historical,
        "nodes": nodes,
        "edges": edges,
        "counts": counts,
    }


def _build_graph(
    conn,
    tenant_id,
    at,
    *,
    include_historical: bool,
    include_episodes: bool,
    scope_list,
    limit: int,
):
    scopes = None if scope_list is None else set(scope_list)
    graph_edges = views.fetch_all_edges(tenant_id, limit)
    graph_edges = views.filter_graph_edges(
        graph_edges, at, include_historical=include_historical
    )
    if scopes is not None:
        graph_edges = [
            edge for edge in graph_edges if views.edge_in_scope(edge, scopes)
        ]

    edges: list[dict] = []
    entity_ids: set[str] = set()
    degree: dict[str, int] = {}

    def bump(node_id: str) -> None:
        degree[node_id] = degree.get(node_id, 0) + 1

    for edge in graph_edges:
        projected = views.edge_view(edge)
        entity_ids.update((projected["source"], projected["target"]))
        edges.append(projected)
        bump(projected["source"])
        bump(projected["target"])

    episode_nodes: list[dict] = []
    source_nodes: list[dict] = []
    episode_total = 0
    if include_episodes:
        episode_edges, episode_nodes, source_nodes, episode_total = _episode_layer(
            conn,
            tenant_id,
            at,
            scope_list,
            min(limit, _ENTRY_GRAPH_LIMIT),
            bump,
        )
        edges.extend(episode_edges)

    entity_meta = views.fetch_entity_meta(tenant_id, entity_ids)
    entity_nodes = [
        views.entity_node_view(
            node_id, name, labels, degree.get(str(node_id), 0), summary
        )
        for node_id, (name, labels, summary) in sorted(entity_meta.items())
    ]
    for node in [*source_nodes, *episode_nodes]:
        node["degree"] = degree.get(node["id"], 0)
    nodes = [*entity_nodes, *source_nodes, *episode_nodes]
    return edges, nodes, {
        "nodes": len(nodes),
        "edges": len(edges),
        "entities": len(entity_nodes),
        "episodes": len(episode_nodes),
        "episodes_total": episode_total,
        "sources": len(source_nodes),
    }


def _label(content: str, max_len: int = 64) -> str:
    first = content.split(". ", 1)[0].strip()
    if len(first) <= max_len:
        return first
    cut = first[: max_len - 1]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "…"


def _source_id(source: str) -> str:
    digest = hashlib.sha256(source.casefold().encode("utf-8")).hexdigest()[:16]
    return f"source:{digest}"


def _episode_layer(conn, tenant_id, at, scope_list, limit, bump):
    params: dict = {"tenant_id": str(tenant_id), "at": at, "limit": limit}
    scope_clause = ""
    if scope_list is not None:
        scope_clause = " AND subject_scope = ANY(CAST(:scopes AS uuid[]))"
        params["scopes"] = scope_list
    rows = conn.execute(
        text(
            "SELECT episode_id, source, source_ref, occurred_at, recorded_at, "
            "content, content_json, meta, legal_hold, count(*) OVER() AS total "
            "FROM episodes WHERE tenant_id=:tenant_id AND occurred_at <= :at"
            f"{scope_clause} "
            "ORDER BY occurred_at DESC, recorded_at DESC, episode_id DESC "
            "LIMIT :limit"
        ),
        params,
    ).mappings().all()

    edges: list[dict] = []
    episodes: list[dict] = []
    sources: dict[str, dict] = {}
    total = int(rows[0]["total"]) if rows else 0
    for row in rows:
        content = row["content"]
        if content is None:
            content = json.dumps(row["content_json"], sort_keys=True, ensure_ascii=False)
        content = content or "Empty record"
        episode_id = f"episode:{row['episode_id']}"
        source_id = _source_id(row["source"])
        episodes.append(
            {
                "id": episode_id,
                "episode_id": str(row["episode_id"]),
                "label": _label(content),
                "type": "episode",
                "node_kind": "episode",
                "content": content,
                "source": row["source"],
                "source_ref": row["source_ref"],
                "occurred_at": _iso(row["occurred_at"]),
                "recorded_at": _iso(row["recorded_at"]),
                "legal_hold": bool(row["legal_hold"]),
                "meta": row["meta"] if isinstance(row["meta"], dict) else {},
            }
        )
        sources.setdefault(
            source_id,
            {
                "id": source_id,
                "label": row["source"].replace("_", " ").title(),
                "type": "source",
                "node_kind": "source",
            },
        )
        edges.append(
            {
                "fact_id": None,
                "fact": "source provenance",
                "source": episode_id,
                "target": source_id,
                "relation": "sourced_from",
                "valid_from": _iso(row["occurred_at"]),
                "valid_to": None,
                "disputed": False,
                "temporal_class": "event",
                "stratum": "provenance",
                "lifecycle": "current",
                "classify_confidence": None,
                "kind": "provenance",
                "genre": None,
            }
        )
        bump(episode_id)
        bump(source_id)
    return edges, episodes, list(sources.values()), total
