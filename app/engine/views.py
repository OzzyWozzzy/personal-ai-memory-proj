"""Read-only projections over the tenant graph for graph, story, and stats APIs.

Everything here reads the tenant's FalkorDB graph (Graphiti is authoritative)
and shapes EntityEdge / Entity rows into the frontend/MCP contract the ledger
endpoints used to serve. The api modules (``app/api/graph.py`` etc.) stay thin:
they own request parsing and the response envelope; this module owns graph
queries and a direct Graphiti edge → view
projection.

Entity edges are native relationships on FalkorDB —
``(Entity)-[e:RELATES_TO]->(Entity)`` — and matching an entity on EITHER
endpoint surfaces object-side edges. Temporal display uses only Graphiti's
stored ``valid_at``/``invalid_at`` fields; no fallback dates or repairs exist.

The sync↔async bridge is :func:`app.engine.graph._run_coro` (as in retrieval):
these functions are called from synchronous FastAPI threadpool handlers, and
every graph coroutine runs on the shared graph event loop.
"""
from __future__ import annotations

from datetime import datetime, timezone

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import get_entity_edge_from_record
from graphiti_core.models.edges.edge_db_queries import get_entity_edge_return_query

from app.engine import graph
from app.engine.native import aware, filter_visible, normalize_relation, validity, value
from app.humanize import friendly_date

_UTC = timezone.utc
_run = graph._run_coro

#: Edge columns projected out of a RELATES_TO match — graphiti's own return
#: projection for this provider, so ``get_entity_edge_from_record`` is parsing
#: exactly the shape it was written against.
_EDGE_RETURN = get_entity_edge_return_query(GraphProvider.FALKORDB)


# --------------------------------------------------------------------------- #
# FalkorDB reads (async, one round trip each) + synchronous wrappers.
# --------------------------------------------------------------------------- #
async def _all_edges_async(engine, limit: int):
    """Every entity-entity edge in the tenant graph (up to ``limit``).

    ``limit`` is inlined as a validated int — it is engine-controlled (never
    user text), so there is no injection surface (same rationale as
    retrieval's fan-out)."""
    cypher = (
        "MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity) "
        "WHERE e.group_id = $gid "
        f"RETURN {_EDGE_RETURN} LIMIT {int(limit)}"
    )
    rows, _, _ = await engine.driver.execute_query(cypher, gid=engine.group_id)
    return [get_entity_edge_from_record(r, GraphProvider.FALKORDB) for r in rows]


async def _entity_edges_async(engine, node_uuid: str, limit: int):
    """All edges touching ``node_uuid`` in BOTH directions (story/history)."""
    cypher = (
        "MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity) "
        "WHERE (n.uuid = $eid OR m.uuid = $eid) AND e.group_id = $gid "
        f"RETURN {_EDGE_RETURN} LIMIT {int(limit)}"
    )
    rows, _, _ = await engine.driver.execute_query(
        cypher, eid=node_uuid, gid=engine.group_id)
    return [get_entity_edge_from_record(r, GraphProvider.FALKORDB) for r in rows]


async def _entity_meta_async(engine, uuids) -> dict[str, tuple[str, list, str]]:
    """uuid → (name, labels, summary) for Entity nodes (one round trip)."""
    if not uuids:
        return {}
    rows, _, _ = await engine.driver.execute_query(
        "MATCH (n:Entity) WHERE n.uuid IN $ids AND n.group_id = $gid "
        "RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels, "
        "n.summary AS summary",
        ids=sorted(uuids), gid=engine.group_id)
    return {
        r["uuid"]: (r["name"], r["labels"] or [], r["summary"] or "")
        for r in rows
    }


async def _entity_one_async(engine, node_uuid: str):
    """(name, labels) for one Entity node, or None if it is not in the graph."""
    rows, _, _ = await engine.driver.execute_query(
        "MATCH (n:Entity {uuid: $eid}) WHERE n.group_id = $gid "
        "RETURN n.name AS name, labels(n) AS labels LIMIT 1",
        eid=str(node_uuid), gid=engine.group_id)
    if not rows:
        return None
    return rows[0]["name"], rows[0]["labels"] or []


async def _fact_counts_async(engine) -> tuple[int, int]:
    """(total edges, currently-valid edges) for the tenant graph."""
    total_rows, _, _ = await engine.driver.execute_query(
        "MATCH ()-[e:RELATES_TO]->() WHERE e.group_id = $gid "
        "RETURN count(e) AS c",
        gid=engine.group_id)
    cur_rows, _, _ = await engine.driver.execute_query(
        "MATCH ()-[e:RELATES_TO]->() WHERE e.group_id = $gid "
        "AND e.invalid_at IS NULL RETURN count(e) AS c",
        gid=engine.group_id)
    return int(total_rows[0]["c"]), int(cur_rows[0]["c"])


def _engine(tenant_id):
    """The tenant's engine, self-configuring the process-wide offline clients on
    first use (mirrors :func:`app.engine.ingest._engine`) so a read that arrives
    before any ingest in this process still works instead of raising. Tests that
    pre-create the engine with explicit clients hit the cache and never reach the
    fallback."""
    try:
        return graph.get_engine(str(tenant_id))
    except RuntimeError:
        from app.engine.ingest import _configure_default_clients
        _configure_default_clients()
        return graph.get_engine(str(tenant_id))


def fetch_all_edges(tenant_id, limit: int):
    """Every entity-entity edge in the tenant graph (synchronous)."""
    return _run(_all_edges_async(_engine(tenant_id), limit))


def fetch_entity_edges(tenant_id, node_uuid, limit: int):
    """One entity's edges, both directions (synchronous)."""
    return _run(_entity_edges_async(_engine(tenant_id), str(node_uuid), limit))


def fetch_entity_meta(tenant_id, uuids) -> dict[str, tuple[str, list, str]]:
    """uuid → (name, labels, summary) for entities (synchronous)."""
    return _run(_entity_meta_async(_engine(tenant_id), set(uuids)))


def fetch_entity(tenant_id, node_uuid):
    """(name, labels) for one entity, or None if absent (synchronous)."""
    return _run(_entity_one_async(_engine(tenant_id), node_uuid))


def fact_counts(tenant_id) -> tuple[int, int]:
    """(facts, facts_current) = (all RELATES_TO edges, those with invalid_at IS
    NULL) for the tenant graph (synchronous)."""
    return _run(_fact_counts_async(_engine(tenant_id)))


# --------------------------------------------------------------------------- #
# Projection helpers (pure).
# --------------------------------------------------------------------------- #
def entity_type_of(labels) -> str:
    """Display type = first non-'Entity' label, else 'thing' (mirrors the ingest
    projection so graph views and the Postgres registry agree)."""
    for label in labels or []:
        if label != "Entity":
            return label
    return "thing"


def _iso(dt) -> str | None:
    dt = aware(dt)
    return dt.isoformat() if isinstance(dt, datetime) else None


def edge_view(edge) -> dict:
    """One entity-entity edge → the frontend graph-edge dict.

    ``disputed`` is always False because there is no separate dispute queue.
    Temporal labelling is projected directly from Graphiti's fields:

    * timeless (no valid_at/invalid_at) → atemporal / semantic / kind 'edge';
    * zero-width window → event / temporal / kind 'event';
    * dated open-or-closed window → state / temporal / kind 'edge'.

    ``lifecycle`` is 'current' while the edge is open (invalid_at IS NULL) else
    'historical'. Graphiti does not provide an epistemic/genre partition.
    """
    valid_at, invalid_at = validity(edge)
    if valid_at is None and invalid_at is None:
        temporal_class, stratum, kind = "atemporal", "semantic", "edge"
    elif valid_at is not None and valid_at == invalid_at:
        temporal_class, stratum, kind = "event", "temporal", "event"
    else:
        temporal_class, stratum, kind = "state", "temporal", "edge"
    lifecycle = "current" if invalid_at is None else "historical"

    return {
        "fact_id": str(value(edge, "uuid")),
        "fact": value(edge, "fact") or "",
        "source": str(value(edge, "source_node_uuid")),
        "target": str(value(edge, "target_node_uuid")),
        "relation": value(edge, "name"),
        "valid_from": _iso(valid_at),
        "valid_to": _iso(invalid_at),
        "disputed": False,
        "temporal_class": temporal_class,
        "stratum": stratum,
        "lifecycle": lifecycle,
        "classify_confidence": None,
        "kind": kind,
        "genre": None,
    }


def entity_node_view(uuid: str, name: str, labels, degree: int,
                     summary: str = "") -> dict:
    """One Entity node → the frontend graph-node dict (node_kind 'entity')."""
    return {
        "id": str(uuid),
        "label": name,
        "type": entity_type_of(labels),
        "node_kind": "entity",
        "degree": degree,
        "summary": summary or "",
    }


# --------------------------------------------------------------------------- #
# Temporal windowing for the graph projection.
# --------------------------------------------------------------------------- #
def filter_graph_edges(edges, at: datetime, *, include_historical: bool):
    """Time-slice the edge set for the graph view.

    Default (``include_historical`` False): as-of ``at`` over Graphiti's actual
    validity fields. That drops closed and future-dated windows.

    ``include_historical`` widens to every window that STARTED by ``at``
    (closed or open), keeping timeless edges — the past graph state a
    time-sliced query wants; the per-edge ``lifecycle`` label then distinguishes
    current from historical. Future-dated windows (start > ``at``) stay
    excluded, matching the ledger projection.
    """
    return filter_visible(edges, at, include_historical=include_historical)


def edge_in_scope(edge, scopes: set[str]) -> bool:
    """ABAC: an edge is visible when it touches an in-scope subject (source OR
    target in ``scopes``) — the same rule retrieval applies."""
    return (str(value(edge, "source_node_uuid")) in scopes
            or str(value(edge, "target_node_uuid")) in scopes)


# --------------------------------------------------------------------------- #
# Story / history rendering.
# --------------------------------------------------------------------------- #
def _story_kind(edge, now: datetime) -> str:
    """The frontend story vocabulary from an edge's window:

    * timeless (no dates) → 'fact';
    * zero-width window → 'event';
    * closed window → 'ended';
    * open, future-dated start → 'began';
    * open, started → 'still-true'.
    """
    start, end = validity(edge)
    if start is None and end is None:
        return "fact"
    if start is not None and start == end:
        return "event"
    if end is not None:
        return "ended"
    if start is not None and start > now:
        return "began"
    return "still-true"


def _story_sort_key(edge):
    """Sort edges chronologically by asserted start (valid_at); timeless edges
    (no start) sort last, tie-broken by uuid for determinism."""
    start, _ = validity(edge)
    start = start or datetime.max.replace(tzinfo=_UTC)
    return (start, str(value(edge, "uuid")))


def story_entries(edges, *, now: datetime | None = None) -> list[dict]:
    """An entity's edges → dated plain-language story entries, oldest first.

    ``text`` is the edge's own fact sentence; ``when`` is the friendly start date
    (empty for timeless facts). Each entry carries a private ``_at`` ISO key (or
    ``""`` for timeless) so callers can merge notes in and re-sort chronologically
    before stripping it (see :func:`app.api.story`)."""
    now = aware(now) or datetime.now(tz=_UTC)
    out = []
    for edge in sorted(edges, key=_story_sort_key):
        start, _ = validity(edge)
        out.append({
            "when": friendly_date(start),
            "text": value(edge, "fact") or "",
            "kind": _story_kind(edge, now),
            "_at": start.isoformat() if start is not None else "",
        })
    return out


def history_entries(edges) -> list[dict]:
    """An entity's edges → JSON-safe edge views, oldest first — the raw record
    behind the story (``/history``)."""
    return [edge_view(e) for e in sorted(edges, key=_story_sort_key)]


def matches_relation(edge, relation: str | None) -> bool:
    """True when ``relation`` is unset or matches the edge's normalized name
    (so ``works_at`` / ``works at`` / ``WORKS_AT`` all match one filter)."""
    if not relation:
        return True
    return normalize_relation(value(edge, "name")) == normalize_relation(relation)
