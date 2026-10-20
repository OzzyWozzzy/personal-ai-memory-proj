"""GET /v1/search — ONE low-friction search box over everything ever entered.

Entity matches, their headline sentences, and their
connections all come from the tenant graph (Graphiti-over-FalkorDB) via
:mod:`app.engine.retrieval`; passages and the raw answer block reuse
:func:`app.engine.retrieval.assemble_context`. The response shape is unchanged
(the frontend + MCP read specific fields):

* **matches** — up to five people/companies/things whose name matches, each a
  plain-language card: ``id`` / ``name`` / ``kind`` / ``sentences`` (current
  facts as sentences) / ``connections`` (``{name, description, other_id}``) /
  ``story_available``;
* **passages** — blended note/document snippets (``text`` / ``when`` / ``source``);
* **answer_block** — the raw facts-route context block (only when a subject
  matched).

``as_of`` is optional natural language ("last March", "2026-02-01"): the search
evaluated at that past moment using Graphiti validity. Every human-facing string comes from
:mod:`app.humanize`.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Request

from app.db import tenant_connection
from graphiti_core.edges import EntityEdge

from app.engine import graph, views
from app.engine.native import validity
from app.engine.retrieval import (
    SEARCH_LIMIT,
    _lexical_overlap,
    _run,
    _search_async,
    _sig_tokens,
    assemble_context,
    extract_mentions,
    native_search_filters,
)
from app.humanize import _sentence, friendly_date, humanize_source, humanize_type
from app.normalize import normalize_time

router = APIRouter()

_UTC = timezone.utc
_MATCH_CAP = 5
_SENTENCE_CAP = 6
_CONNECTION_CAP = 8
_PASSAGE_CAP = 6
_PASSAGE_BUDGET = 700


@router.get("/v1/search")
def search(request: Request, q: str = Query(default=""),
           as_of: str | None = Query(default=None)) -> dict:
    ctx = request.state.tenant_ctx
    tid = ctx.tenant_id
    q = (q or "").strip()
    if not q:
        return {"query": q, "as_of": None, "matches": [], "passages": [],
                "answer_block": None}

    now = datetime.now(tz=_UTC)
    at = None
    if as_of and as_of.strip():
        resolved, unresolved = normalize_time(
            as_of, occurred_at=now, timezone=ctx.timezone, locale=ctx.locale)
        if unresolved:
            raise HTTPException(
                422, f"I couldn't understand “{as_of.strip()}” as a date — "
                     f"try something like “last March” or “2026-05-01”.")
        at = resolved

    scopes = getattr(ctx, "allowed_subject_scopes", None)
    scope_set = None if scopes is None else {str(s) for s in scopes}
    render_at = at or now

    # Self-configuring fallback (mirrors views/ingest): a fresh process whose
    # FIRST request is /v1/search must not 500 for lack of configured clients.
    engine = views._engine(tid)

    # -- Resolve matching entities + fan out each one's edges (one graph phase). --
    async def _phase():
        results = await _search_async(
            engine, q, "COMBINED_HYBRID_SEARCH_CROSS_ENCODER", SEARCH_LIMIT,
            filters=native_search_filters(
                as_of=at, as_believed_at=None, include_historical=False))
        nodes = list(results.nodes)
        # Prefer nodes whose NAME lexically matches a proper-noun mention; fall
        # back to the ranked hits for an all-lowercase / descriptive query.
        mentions = extract_mentions(q)
        if mentions:
            toks: set[str] = set()
            for m in mentions:
                toks |= _sig_tokens(m)
            preferred = [n for n in nodes if _lexical_overlap(n.name, toks)]
            nodes = preferred or nodes
        if scope_set is not None:
            nodes = [n for n in nodes if str(n.uuid) in scope_set]
        nodes = nodes[:_MATCH_CAP]

        cards = []
        for node in nodes:
            # Graphiti endpoint lookup; natively endpoint-neutral on FalkorDB
            # (the undirected incident-edge MATCH in edges.py).
            edges = await EntityEdge.get_by_node_uuid(engine.driver, str(node.uuid))
            other_ids = {
                (str(e.target_node_uuid) if str(e.source_node_uuid) == str(node.uuid)
                 else str(e.source_node_uuid))
                for e in edges
            }
            rows, _, _ = await engine.driver.execute_query(
                "MATCH (n:Entity) WHERE n.uuid IN $ids AND n.group_id = $gid "
                "RETURN n.uuid AS uuid, n.name AS name",
                ids=sorted(other_ids | {str(node.uuid)}), gid=engine.group_id)
            names = {row["uuid"]: row["name"] for row in rows}
            cards.append((node, edges, names))
        return cards

    cards = _run(_phase())

    matches = [_entity_card(node, edges, names, render_at, scope_set)
               for node, edges, names in cards]

    with tenant_connection(tid) as conn:
        passages = _passages(conn, tid, q, scopes)

        answer_block = None
        if matches:
            facts_resp = assemble_context(
                conn, tid, query=q, include="facts", fmt="block", as_of=at,
                actor="search", allowed_subject_scopes=scopes)
            answer_block = facts_resp.get("block") or None

    return {
        "query": q,
        "as_of": friendly_date(at) if at else None,
        "matches": matches,
        "passages": passages,
        "answer_block": answer_block,
    }


# ---------------------------------------------------------------------------
# Entity cards
# ---------------------------------------------------------------------------
def _friendly_window(edge) -> str:
    """The window grammar in human words: ``since May 2, 2026`` / ``on May 2, 2026``
    / ``May 2, 2026 – Jun 1, 2026`` / ``""`` (timeless)."""
    start_at, end_at = validity(edge)
    if start_at is None and end_at is None:
        return ""
    start = friendly_date(start_at) if start_at is not None else "an unknown date"
    if end_at is None:
        return f"since {start}"
    if start_at is not None and start_at == end_at:
        return f"on {start}"
    return f"{start} – {friendly_date(end_at)}"


def _sentence_for(edge) -> str:
    window = _friendly_window(edge)
    text = f"{edge.fact} ({window})" if window else edge.fact
    return _sentence(text)


def _entity_card(node, edges, names, render_at, scope_set) -> dict:
    nid = str(node.uuid)
    # Only currently-valid edges (as of the render instant) drive the card.
    current = views.filter_graph_edges(edges, render_at, include_historical=False)
    if scope_set is not None:
        current = [e for e in current
                   if str(e.source_node_uuid) in scope_set
                   or str(e.target_node_uuid) in scope_set]

    # Headline sentences: this entity's subject-side edges, dated first.
    subject_edges = [e for e in current if str(e.source_node_uuid) == nid]
    subject_edges.sort(
        key=lambda e: (validity(e)[0] is None,
                       validity(e)[0] or datetime.min.replace(tzinfo=_UTC)),
        reverse=True)
    sentences = [_sentence_for(e) for e in subject_edges[:_SENTENCE_CAP]]

    # Connections: entity↔entity edges, both directions, deduped per neighbour.
    connections, seen = [], set()
    for e in current:
        other = (str(e.target_node_uuid) if str(e.source_node_uuid) == nid
                 else str(e.source_node_uuid))
        if other == nid or other in seen:
            continue
        other_name = names.get(other)
        if not other_name:
            continue
        seen.add(other)
        connections.append({
            "other_id": other,
            "name": other_name,
            "description": f"Connected to {other_name} — {_sentence_for(e)}",
        })
        if len(connections) >= _CONNECTION_CAP:
            break

    kind_label = next((l for l in getattr(node, "labels", []) if l != "Entity"), None)
    story_available = any(any(validity(e)) for e in edges)

    return {
        "id": nid,
        "name": node.name,
        "kind": humanize_type(kind_label),
        "sentences": sentences,
        "connections": connections,
        "story_available": story_available,
    }


# ---------------------------------------------------------------------------
# Passages (knowledge route, re-rendered plainly)
# ---------------------------------------------------------------------------
def _passages(conn, tid, q, scopes) -> list[dict]:
    resp = assemble_context(
        conn, tid, query=q, include="knowledge", fmt="json",
        token_budget=_PASSAGE_BUDGET, actor="search",
        allowed_subject_scopes=scopes)
    out = []
    for item in resp.get("items", [])[:_PASSAGE_CAP]:
        occurred = item.get("occurred_at")
        when = friendly_date(datetime.fromisoformat(occurred)) if occurred else None
        out.append({
            "text": item.get("snippet") or "",
            "when": when,
            "source": humanize_source(item.get("source")),
        })
    return out
