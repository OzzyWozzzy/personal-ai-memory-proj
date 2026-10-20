"""The single Graphiti-native write path for the memory engine.

Every write enters through ``/v1/remember`` and funnels through here. One text
episode in, one :class:`IngestReport` out. The pipeline, in order:

1. **Postgres episode first** (:func:`app.episodes.ingest_episode`): immutable
   ground truth, content-hash deduped. A deduped episode SKIPS graph ingestion
   entirely — replays must never double-write edges into the graph.
2. **Graphiti extraction** (``graphiti.add_episode``): entity/edge extraction,
   dedupe, native validity assignment and contradiction/invalidation handling.
   The application does not mutate returned edges afterward.
3. **Entity projection**: upsert each graph node into the Postgres
   ``entities`` display registry for search and visualization — one-way and
   graph-authoritative.

The graph half (steps 2-3) is exposed as :func:`ingest_episode_into_graph` for
rebuilding graph state from the immutable episode ledger.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from app.audit import write_audit
from app.db import tenant_connection, tenant_graph_write_lock
from app.episodes import ingest_episode
from app.engine import graph as _graph

# app.engine is imported (transitively via app.engine.graph) before graphiti_core,
# pinning EMBEDDING_DIM=384. Import graphiti_core only after that.
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodeType


# --------------------------------------------------------------------------- #
# Custom entity ontology
# --------------------------------------------------------------------------- #
# Graphiti injects each type's docstring into the extraction prompt as its
# classification description, so obvious fiction is UNDERSTOOD rather than
# stored as literal real-world fact: "Bilbo found the ring" yields entities
# labeled FictionalCharacter/CreativeWork instead of a person named Bilbo.
# The labels flow into node.labels, the Postgres entity projection, retrieval
# output, and the graph view's coloring. Everything else keeps Graphiti's
# default classification.
class FictionalCharacter(BaseModel):
    """A character from a work of fiction — a novel, film, TV series, game,
    comic, or myth (e.g. Bilbo Baggins, Darth Vader) — NOT a real person the
    user knows or works with."""


class CreativeWork(BaseModel):
    """A named creative work or media title — a book, novel, film, TV series,
    game, album, or similar (e.g. The Hobbit, Star Wars) — NOT a company,
    project, or product the user works on."""


_ENTITY_TYPES: dict[str, type[BaseModel]] = {
    "FictionalCharacter": FictionalCharacter,
    "CreativeWork": CreativeWork,
}

_AUDIT_ACTOR = "engine.ingest"

# Postgres episode ``source`` -> Graphiti EpisodeType. ``message`` expects
# explicit ``actor: content`` dialogue framing, so only chat uses it. Gmail's
# normalized external text is unstructured; only chat uses message framing.
# Unknown sources default to text.
_EPISODE_TYPE: dict[str, EpisodeType] = {
    "note": EpisodeType.text,
    "doc": EpisodeType.text,
    "manual_fact": EpisodeType.text,
    "chat": EpisodeType.message,
    "email": EpisodeType.text,
    "json_event": EpisodeType.json,
}


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass
class IngestReport:
    """The outcome of one episode's ingestion.

    ``edges`` are the NEW entity edges this episode contributed;
    ``invalidated_edges`` are pre-existing edges Graphiti's contradiction and
    temporal resolution closed. No application repair pass follows Graphiti.
    """

    episode_id: str
    deduped: bool
    edges: list[EntityEdge] = field(default_factory=list)
    invalidated_edges: list[EntityEdge] = field(default_factory=list)
    nodes: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class EpisodeGraphState:
    """Idempotency state for one immutable Postgres episode.

    ``fact_count`` is the number of graph assertions carrying this episode's
    Graphiti UUID as provenance.  A present episode with zero facts is a valid
    outcome (for example, prose that the extractor judged to contain no
    assertion), not evidence that ingestion failed.
    """

    present: bool
    fact_count: int = 0


# --------------------------------------------------------------------------- #
# sync <-> async bridge — delegated to the graph loop, which owns ALL graph I/O
# (redis.asyncio connections are loop-bound; per-call asyncio.run loops would
# corrupt the shared client pool).
# --------------------------------------------------------------------------- #
def _run_coro(coro):
    return _graph._run_coro(coro)  # noqa: SLF001 — the one sanctioned sync bridge


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _tenant_id(tenant_ctx_or_id) -> str:
    """Accept a tenant context (``.tenant_id``) or a bare tenant id."""
    return str(getattr(tenant_ctx_or_id, "tenant_id", tenant_ctx_or_id))


def flatten_json_text(payload) -> str:
    """One flat sentence describing a JSON payload, for graph extraction.

    Mirrors the pre-v4 remember behavior: scalar key/value pairs joined with
    "; ", falling back to canonical JSON. Shared by the remember json path and
    the rebuild command so both render a json_event episode identically.
    """
    if isinstance(payload, dict):
        pairs = [f"{k}: {v}" for k, v in payload.items()
                 if isinstance(v, (str, int, float, bool))]
        if pairs:
            return "; ".join(pairs)
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _episode_type(source: str) -> EpisodeType:
    return _EPISODE_TYPE.get(source, EpisodeType.text)


def _entity_type_of(node) -> str:
    """Projection entity_type: the first non-'Entity' label, else 'thing'."""
    for label in getattr(node, "labels", None) or []:
        if label != "Entity":
            return label
    return "thing"


def _engine(tenant_id: str):
    """The tenant's engine, self-configuring the process-wide clients on first
    use so the write path works without a separate bootstrap call. Tests that
    call ``graph.configure_clients`` first win (we only default when unset)."""
    try:
        engine = _graph.get_engine(tenant_id)
    except RuntimeError:
        _configure_default_clients()
        engine = _graph.get_engine(tenant_id)
    # (The vendor's missing ``driver._database`` init is fixed canonically in
    # TenantEngine.__init__ — engines arrive here group-routed already.)
    return engine


_clients_lock = threading.Lock()


def _configure_default_clients() -> None:
    """Register the offline Graphiti clients as the process-wide default."""
    with _clients_lock:
        # Re-check under the lock: another thread may have configured already.
        if _graph._clients is not None:  # noqa: SLF001 — intentional module cooperation
            return
        from app.engine.bridge import get_llm_client
        from app.engine.clients import FastEmbedEmbedder, LateInteractionReranker

        _graph.configure_clients(
            llm_client=get_llm_client(),
            embedder=FastEmbedEmbedder(),
            cross_encoder=LateInteractionReranker(),
        )


# --------------------------------------------------------------------------- #
# Public write path
# --------------------------------------------------------------------------- #
def ingest_text(
    tenant_ctx_or_id,
    *,
    text: str,
    occurred_at: datetime,
    source: str,
    meta: dict | None = None,
) -> IngestReport:
    """Store one text episode and fold it into the graph. See module docstring.

    ``source`` is the Postgres episode source ('note'/'doc'/'json_event'/...);
    the Graphiti EpisodeType is derived from it. A deduped episode returns a
    report with ``deduped=True`` and no graph work performed.
    """
    tenant_id = _tenant_id(tenant_ctx_or_id)
    with tenant_connection(tenant_id) as conn:
        episode_id, deduped = ingest_episode(
            conn, tenant_id, source=source, occurred_at=occurred_at,
            content=text, meta=meta or None)
    if deduped:
        return IngestReport(episode_id=str(episode_id), deduped=True)
    return ingest_episode_into_graph(
        tenant_id, str(episode_id), text=text, occurred_at=occurred_at,
        source=source, meta=meta)


def ingest_episode_into_graph(
    tenant_ctx_or_id,
    episode_id: str,
    *,
    text: str,
    occurred_at: datetime,
    source: str,
    meta: dict | None = None,
) -> IngestReport:
    """The graph half (steps 2-5) for an episode already stored in Postgres.

    Used by the document/json endpoints and the rebuild command, which own the
    episode row themselves. Runs native ``add_episode`` and entity projection.
    """
    tenant_id = _tenant_id(tenant_ctx_or_id)
    engine = _engine(tenant_id)
    # Same-tenant graph writes are serialized ACROSS REPLICAS: Graphiti's
    # entity dedupe reads the current graph, so two concurrent add_episode
    # calls for one tenant could each miss the other's in-flight nodes.
    with tenant_graph_write_lock(tenant_id):
        result = _run_coro(_graph_ingest_async(
            engine, name=str(episode_id), text=text, occurred_at=occurred_at,
            source=source))

    return _finish_graph_ingest(tenant_id, str(episode_id), result)


def ensure_episode_in_graph(
    tenant_ctx_or_id,
    episode_id: str,
    *,
    text: str,
    occurred_at: datetime,
    source: str,
    meta: dict | None = None,
) -> IngestReport:
    """Fold an existing episode into Graphiti exactly once.

    Unlike a check followed by :func:`ingest_episode_into_graph`, the presence
    check and graph write share the cross-replica tenant graph lock. This is the
    recovery-safe path for request handlers which commit the immutable episode
    before extraction: a retry repairs a missing graph write, while concurrent
    retries cannot both submit the same episode to Graphiti.

    ``IngestReport.deduped`` is true when the graph episode already existed.
    ``meta`` is accepted for parity with the other write helpers; Graphiti's
    native text extraction currently consumes the normalized text/source/time.
    """
    del meta  # Reserved for parity with ingest_episode_into_graph.
    tenant_id = _tenant_id(tenant_ctx_or_id)
    episode_id = str(episode_id)
    engine = _engine(tenant_id)
    existing_nodes = None
    with tenant_graph_write_lock(tenant_id):
        state = _episode_graph_state(engine, episode_id)
        if state.present:
            # Graphiti may have committed immediately before a transient
            # Postgres projection failure. Read the episode's authoritative
            # mentions so an exact retry can repair the display registry.
            existing_nodes = _run_coro(
                _episode_entities_async(engine, episode_id)
            )
        else:
            result = _run_coro(_graph_ingest_async(
                engine, name=episode_id, text=text, occurred_at=occurred_at,
                source=source))

    if existing_nodes is not None:
        with tenant_connection(tenant_id) as conn:
            _project_entities(conn, tenant_id, existing_nodes)
        return IngestReport(episode_id=episode_id, deduped=True)

    return _finish_graph_ingest(tenant_id, episode_id, result)


def _finish_graph_ingest(
    tenant_id: str,
    episode_id: str,
    result: dict,
) -> IngestReport:
    """Project one completed Graphiti write into Postgres and render a report."""

    # Postgres side: graph-authoritative display projection. One connection.
    with tenant_connection(tenant_id) as conn:
        for edge in result["invalidated_edges"]:
            write_audit(conn, tenant_id, _AUDIT_ACTOR, "graph.edge_invalidated", {
                "edge_uuid": str(edge.uuid),
                "fact": edge.fact,
                "invalid_at": edge.invalid_at.isoformat() if edge.invalid_at else None,
                "expired_at": edge.expired_at.isoformat() if edge.expired_at else None,
                "episode_id": str(episode_id),
            })
        _project_entities(conn, tenant_id, result["nodes"])

    return IngestReport(
        episode_id=str(episode_id),
        deduped=False,
        edges=result["new_edges"],
        invalidated_edges=result["invalidated_edges"],
        nodes=result["nodes"],
    )


# --------------------------------------------------------------------------- #
# Graph work (async)
# --------------------------------------------------------------------------- #
async def _graph_ingest_async(engine, *, name, text, occurred_at, source) -> dict:
    results = await engine.graphiti.add_episode(
        name=name,
        episode_body=text,
        source_description=source,
        reference_time=occurred_at,
        source=_episode_type(source),
        group_id=engine.group_id,
        entity_types=_ENTITY_TYPES,
    )

    episode_uuid = results.episode.uuid
    all_edges = list(results.edges)  # = resolved (new) + invalidated (predecessors)

    # New vs invalidated: a resolved edge carries this episode's uuid in its
    # ``episodes`` list; an invalidated predecessor was only closed, not touched.
    new_edges = [e for e in all_edges if episode_uuid in (e.episodes or [])]
    invalidated_edges = [e for e in all_edges
                         if episode_uuid not in (e.episodes or [])]

    return {
        "new_edges": new_edges,
        "invalidated_edges": invalidated_edges,
        "nodes": list(results.nodes),
    }


async def _episode_entities_async(engine, episode_id: str) -> list[EntityNode]:
    """Load the current entities mentioned by an existing graph episode."""
    rows, _, _ = await engine.driver.execute_query(
        "MATCH (episode:Episodic {name: $name})-[:MENTIONS]->(node:Entity) "
        "RETURN DISTINCT node.uuid AS uuid",
        name=str(episode_id),
    )
    uuids = [str(row["uuid"]) for row in rows]
    if not uuids:
        return []
    return await EntityNode.get_by_uuids(
        engine.driver, uuids, group_id=engine.group_id
    )


def episode_graph_state(
    tenant_ctx_or_id,
    episode_id: str,
) -> EpisodeGraphState:
    """Return graph presence and assertion count for a Postgres episode.

    Graph episodes are named after the immutable Postgres ``episode_id``.
    Callers use this check immediately before graph ingestion so a retry after
    a Postgres commit (or after a lost HTTP response) cannot double-write graph
    edges. Counting provenance-bearing facts also lets ingestion retries
    distinguish a legitimate no-fact extraction from a missing graph write.
    """
    engine = _engine(_tenant_id(tenant_ctx_or_id))
    return _episode_graph_state(engine, str(episode_id))


def _episode_graph_state(engine, episode_id: str) -> EpisodeGraphState:
    """Read graph presence using an already-resolved tenant engine."""
    async def _check() -> EpisodeGraphState:
        rows, _, _ = await engine.driver.execute_query(
            "MATCH (e:Episodic {name: $name}) RETURN e.uuid AS uuid LIMIT 1",
            name=str(episode_id))
        if not rows:
            return EpisodeGraphState(present=False)
        episode_uuid = str(rows[0]["uuid"])
        fact_rows, _, _ = await engine.driver.execute_query(
            "MATCH ()-[e:RELATES_TO]->() WHERE $uuid IN e.episodes "
            "RETURN count(e) AS count",
            uuid=episode_uuid,
        )
        count = int(fact_rows[0]["count"]) if fact_rows else 0
        return EpisodeGraphState(present=True, fact_count=count)

    return _run_coro(_check())


def episode_in_graph(tenant_ctx_or_id, episode_id: str) -> bool:
    """Compatibility predicate for idempotent re-extraction endpoints."""

    return episode_graph_state(tenant_ctx_or_id, episode_id).present


# --------------------------------------------------------------------------- #
# Entity projection (Postgres display registry — plan §3.7)
# --------------------------------------------------------------------------- #
def _project_entities(conn, tenant_id: str, nodes) -> None:
    """Upsert each graph node into the Postgres ``entities`` table.

    Graph-authoritative, one-way. entity_id = node.uuid, canonical_name =
    node.name, entity_type = first non-'Entity' label (else 'thing'). On a name
    or type change the display row follows the graph.
    """
    from sqlalchemy import text as _sql

    for node in nodes:
        conn.execute(
            _sql(
                "INSERT INTO entities (entity_id, tenant_id, entity_type,"
                " canonical_name)"
                " VALUES (CAST(:eid AS uuid), :tid, :etype, :name)"
                " ON CONFLICT (entity_id) DO UPDATE SET"
                " canonical_name = EXCLUDED.canonical_name,"
                " entity_type = EXCLUDED.entity_type"
            ),
            {"eid": str(node.uuid), "tid": str(tenant_id),
             "etype": _entity_type_of(node), "name": node.name},
        )
