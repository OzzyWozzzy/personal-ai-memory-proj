"""Tests for app/engine/views.py + the projection endpoints it feeds
(/v1/graph, /v1/subjects/{id}/story + /history, /v1/stats, /v1/subjects/{id}/profile).

Graph state is built DIRECTLY (EntityNode.save / EntityEdge.save on the tenant
engine's driver) — no LLM: the stub LLM client raises on any call. Embeddings are
REAL (FastEmbedEmbedder) only because save requires them; the views are pure graph
reads. Postgres is used for the Notes layer + episode/note counts. Each test
sandboxes MEMORY_GRAPH_DIR (autouse fixture) and tears the engine down.
"""
from __future__ import annotations

# Import-ordering contract: app.engine sets EMBEDDING_DIM=384 before graphiti_core.
import app.engine  # noqa: F401
import app.engine.graph as graph

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import text

from graphiti_core.edges import EntityEdge
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.nodes import EntityNode

from app.engine import views
from app.engine.clients import FastEmbedEmbedder, PassthroughCrossEncoder
from app.api.graph import _build_graph

_UTC = timezone.utc


def D(y, m, d):
    return datetime(y, m, d, tzinfo=_UTC)


class StubLLMClient(LLMClient):
    """Must never be invoked: the views are LLM-free by construction."""

    def __init__(self):
        super().__init__(config=None)

    async def _generate_response(self, messages, response_model=None,
                                 max_tokens=None, model_size=None):
        raise AssertionError("views must not call the LLM")


EMBEDDER = FastEmbedEmbedder()


def _clients():
    return dict(llm_client=StubLLMClient(), embedder=EMBEDDER,
                cross_encoder=PassthroughCrossEncoder())


@pytest.fixture(autouse=True)
def _sandbox_graph_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_GRAPH_DIR", str(tmp_path / "graphdb"))
    yield
    graph.close_all()


# --------------------------------------------------------------------------- #
# Direct graph seeding (no LLM).
# --------------------------------------------------------------------------- #
def _node(engine, name, *, summary="", labels=("Person",)):
    return EntityNode(uuid=str(uuid4()), name=name, group_id=engine.group_id,
                      labels=list(labels), summary=summary)


def _edge(engine, src, tgt, name, fact, *, valid_at=None, invalid_at=None,
          attributes=None, episodes=("ep1",)):
    e = EntityEdge(
        uuid=str(uuid4()), source_node_uuid=src.uuid, target_node_uuid=tgt.uuid,
        name=name, fact=fact, group_id=engine.group_id,
        created_at=D(2024, 1, 1), valid_at=valid_at, invalid_at=invalid_at,
        episodes=list(episodes))
    if attributes is not None:
        e.attributes = attributes
    return e


def _seed(engine, nodes, edges):
    async def _go():
        for n in nodes:
            await n.generate_name_embedding(EMBEDDER)
            await n.save(engine.driver)
        for e in edges:
            await e.generate_embedding(EMBEDDER)
            await e.save(engine.driver)

    graph._run_coro(_go())


def _engine_for(tenant):
    return graph.get_engine(str(tenant.tenant_id), **_clients())


_GRAPH_EDGE_KEYS = {"fact_id", "fact", "source", "target", "relation", "valid_from",
                    "valid_to", "disputed", "temporal_class", "stratum",
                    "lifecycle", "classify_confidence", "kind", "genre"}
_GRAPH_NODE_KEYS = {"id", "label", "type", "node_kind", "degree", "summary"}


# --------------------------------------------------------------------------- #
# (1) edge_view projection — the deterministic edge → dict mapping (pure).
# --------------------------------------------------------------------------- #
def test_edge_view_temporal_classes():
    n1 = EntityNode(uuid=str(uuid4()), name="A", group_id="g")
    n2 = EntityNode(uuid=str(uuid4()), name="B", group_id="g")

    def mk(valid_at=None, invalid_at=None, attributes=None):
        e = EntityEdge(uuid=str(uuid4()), source_node_uuid=n1.uuid,
                       target_node_uuid=n2.uuid, name="rel", fact="A rel B",
                       group_id="g", created_at=D(2024, 1, 1),
                       valid_at=valid_at, invalid_at=invalid_at, episodes=["ep"])
        if attributes is not None:
            e.attributes = attributes
        return e

    # timeless → atemporal / semantic / edge / current
    v = views.edge_view(mk())
    assert set(v) == _GRAPH_EDGE_KEYS
    assert (v["temporal_class"], v["stratum"], v["kind"], v["lifecycle"]) == \
        ("atemporal", "semantic", "edge", "current")
    assert v["valid_from"] is None and v["valid_to"] is None
    assert v["disputed"] is False and v["classify_confidence"] is None
    assert v["genre"] is None
    assert v["fact"] == "A rel B"

    # dated open → state / temporal / edge / current
    v = views.edge_view(mk(valid_at=D(2024, 3, 1)))
    assert (v["temporal_class"], v["stratum"], v["kind"], v["lifecycle"]) == \
        ("state", "temporal", "edge", "current")
    assert v["valid_from"] == D(2024, 3, 1).isoformat()

    # dated closed → state / temporal / edge / historical
    v = views.edge_view(mk(valid_at=D(2024, 3, 1), invalid_at=D(2024, 5, 1)))
    assert (v["temporal_class"], v["lifecycle"]) == ("state", "historical")
    assert v["valid_to"] == D(2024, 5, 1).isoformat()

    # zero-width → event / temporal / event
    v = views.edge_view(mk(valid_at=D(2024, 3, 1), invalid_at=D(2024, 3, 1)))
    assert (v["temporal_class"], v["stratum"], v["kind"]) == \
        ("event", "temporal", "event")

    # Custom epistemic attributes are not interpreted in Graphiti-native mode.
    v = views.edge_view(mk(attributes={"epistemic": "in_world"}))
    assert v["genre"] is None


# --------------------------------------------------------------------------- #
# (2) /v1/graph builder: node/edge/count shape from real graph state.
# --------------------------------------------------------------------------- #
def test_build_graph_shape_and_counts(make_tenant, tenant_conn):
    t = make_tenant()
    engine = _engine_for(t)
    marisol = _node(engine, "Marisol Quintero", labels=("Person",))
    porto = _node(engine, "Porto", labels=("Place",))
    faro = _node(engine, "Faro", labels=("Thing",))
    open_dated = _edge(engine, marisol, porto, "lives_in",
                       "Marisol Quintero lives in Porto", valid_at=D(2024, 1, 1))
    timeless = _edge(engine, marisol, faro, "owns",
                     "Marisol Quintero owns a greyhound named Faro")
    _seed(engine, [marisol, porto, faro], [open_dated, timeless])

    with tenant_conn(t.tenant_id) as conn:
        edges, nodes, counts = _build_graph(
            conn, t.tenant_id, D(2025, 1, 1),
            include_historical=False, include_episodes=False,
            scope_list=None, limit=500)

    assert counts == {"nodes": 3, "edges": 2, "entities": 3,
                      "episodes": 0, "episodes_total": 0, "sources": 0}
    for e in edges:
        assert set(e) == _GRAPH_EDGE_KEYS
        assert e["disputed"] is False
    for n in nodes:
        assert set(n) == _GRAPH_NODE_KEYS
        assert n["node_kind"] == "entity"
    # Marisol has degree 2 (two edges); label + type projected from the graph.
    by_id = {n["id"]: n for n in nodes}
    assert by_id[marisol.uuid]["label"] == "Marisol Quintero"
    assert by_id[marisol.uuid]["type"] == "Person"
    assert by_id[marisol.uuid]["degree"] == 2
    assert by_id[marisol.uuid]["summary"] == ""
    assert by_id[porto.uuid]["type"] == "Place"


# --------------------------------------------------------------------------- #
# (3) Temporal filtering: as_of hides closed/future; include_historical widens.
# --------------------------------------------------------------------------- #
def _seed_ada_moves(engine):
    ada = _node(engine, "Ada Vermeer")
    utrecht = _node(engine, "Utrecht", labels=("Place",))
    lisbon = _node(engine, "Lisbon", labels=("Place",))
    closed = _edge(engine, ada, utrecht, "lives_in", "Ada Vermeer lives in Utrecht",
                   valid_at=D(2023, 1, 1), invalid_at=D(2024, 6, 1))
    open_ = _edge(engine, ada, lisbon, "lives_in", "Ada Vermeer lives in Lisbon",
                  valid_at=D(2024, 6, 1))
    _seed(engine, [ada, utrecht, lisbon], [closed, open_])
    return ada, closed, open_


def _fact_ids(edges):
    return {e["fact_id"] for e in edges}


def test_build_graph_temporal_filtering(make_tenant, tenant_conn):
    t = make_tenant()
    engine = _engine_for(t)
    _ada, closed, open_ = _seed_ada_moves(engine)

    with tenant_conn(t.tenant_id) as conn:
        # After the move: only the open Lisbon window is current.
        after, _n, _c = _build_graph(conn, t.tenant_id, D(2025, 1, 1),
                                     include_historical=False,
                                     include_episodes=False, scope_list=None,
                                     limit=500)
        # Before the move: only the (then-open) Utrecht window covers the instant.
        before, _n, _c = _build_graph(conn, t.tenant_id, D(2023, 6, 1),
                                      include_historical=False,
                                      include_episodes=False, scope_list=None,
                                      limit=500)
        # Historical: both windows started by now → both visible.
        hist, _n, _c = _build_graph(conn, t.tenant_id, D(2025, 1, 1),
                                    include_historical=True,
                                    include_episodes=False, scope_list=None,
                                    limit=500)

    assert _fact_ids(after) == {open_.uuid}
    assert _fact_ids(before) == {closed.uuid}
    assert _fact_ids(hist) == {closed.uuid, open_.uuid}
    # lifecycle labels the historical projection.
    lc = {e["fact_id"]: e["lifecycle"] for e in hist}
    assert lc[closed.uuid] == "historical" and lc[open_.uuid] == "current"


# --------------------------------------------------------------------------- #
# (5) ABAC scope filtering: out-of-scope edges dropped.
# --------------------------------------------------------------------------- #
def test_build_graph_abac_scope(make_tenant, tenant_conn):
    t = make_tenant()
    engine = _engine_for(t)
    rosa = _node(engine, "Rosa Marin")
    labs = _node(engine, "Kolibri Labs", labels=("Organization",))
    tomas = _node(engine, "Tomas Berg")
    in_scope = _edge(engine, rosa, labs, "works_at",
                     "Rosa Marin works at Kolibri Labs", valid_at=D(2024, 1, 1))
    out_scope = _edge(engine, tomas, labs, "invests_in",
                      "Tomas Berg invests in Kolibri Labs", valid_at=D(2024, 2, 1))
    _seed(engine, [rosa, labs, tomas], [in_scope, out_scope])

    with tenant_conn(t.tenant_id) as conn:
        edges, nodes, _c = _build_graph(
            conn, t.tenant_id, D(2025, 1, 1), include_historical=False,
            include_episodes=False, scope_list=[rosa.uuid], limit=500)

    assert _fact_ids(edges) == {in_scope.uuid}
    node_ids = {n["id"] for n in nodes}
    assert rosa.uuid in node_ids and labs.uuid in node_ids
    assert tomas.uuid not in node_ids  # only participates in the dropped edge


# --------------------------------------------------------------------------- #
# (6) Story: entry-kind vocabulary (pure) + ordering through the endpoint.
# --------------------------------------------------------------------------- #
def test_story_entry_kinds():
    n = EntityNode(uuid=str(uuid4()), name="Subj", group_id="g")

    def mk(valid_at=None, invalid_at=None):
        return EntityEdge(uuid=str(uuid4()), source_node_uuid=n.uuid,
                          target_node_uuid=str(uuid4()), name="rel", fact="f",
                          group_id="g", created_at=D(2024, 1, 1),
                          valid_at=valid_at, invalid_at=invalid_at, episodes=["e"])

    now = D(2024, 6, 1)
    edges = [
        mk(valid_at=D(2024, 1, 1)),                       # still-true (open, past)
        mk(valid_at=D(2025, 1, 1)),                       # began (open, future)
        mk(valid_at=D(2023, 1, 1), invalid_at=D(2023, 6, 1)),  # ended (closed)
        mk(valid_at=D(2024, 2, 2), invalid_at=D(2024, 2, 2)),  # event (zero-width)
        mk(),                                             # fact (timeless)
    ]
    kinds = [e["kind"] for e in views.story_entries(edges, now=now)]
    assert set(kinds) == {"still-true", "began", "ended", "event", "fact"}
    # Every entry carries {when, text, kind} (+ the internal _at sort carrier).
    for e in views.story_entries(edges, now=now):
        assert set(e) == {"when", "text", "kind", "_at"}


def test_story_endpoint(client, make_tenant):
    graph.configure_clients(**_clients())
    t = make_tenant()
    engine = _engine_for(t)
    ada, closed, open_ = _seed_ada_moves(engine)
    hdr = {"X-API-Key": t.api_key}
    resp = client.get(f"/v1/subjects/{ada.uuid}/story", headers=hdr)
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Ada Vermeer"
    assert body["kind"]  # a humanized type label
    kinds = [e["kind"] for e in body["story"]]
    assert "ended" in kinds       # Utrecht window closed
    assert "still-true" in kinds  # Lisbon window open
    # Chronological: Utrecht (2023) before Lisbon (2024).
    texts = [e["text"] for e in body["story"]]
    assert texts.index("Ada Vermeer lives in Utrecht") < \
        texts.index("Ada Vermeer lives in Lisbon")

    # Unknown subject → 404.
    missing = client.get(f"/v1/subjects/{uuid4()}/story", headers=hdr)
    assert missing.status_code == 404


# --------------------------------------------------------------------------- #
# (7) History: same edge set as JSON, optional relation filter.
# --------------------------------------------------------------------------- #
def test_history_endpoint(client, make_tenant):
    graph.configure_clients(**_clients())
    t = make_tenant()
    engine = _engine_for(t)
    ada, closed, open_ = _seed_ada_moves(engine)
    other = _node(engine, "Rowing Club", labels=("Organization",))
    member = _edge(engine, ada, other, "member_of", "Ada Vermeer is a member of the Rowing Club")
    _seed(engine, [other], [member])

    hdr = {"X-API-Key": t.api_key}
    full = client.get(f"/v1/subjects/{ada.uuid}/history", headers=hdr).json()
    assert full["relation"] is None
    assert _fact_ids(full["history"]) == {closed.uuid, open_.uuid, member.uuid}

    filtered = client.get(f"/v1/subjects/{ada.uuid}/history",
                          headers=hdr, params={"relation": "lives_in"}).json()
    assert filtered["relation"] == "lives_in"
    assert _fact_ids(filtered["history"]) == {closed.uuid, open_.uuid}


# --------------------------------------------------------------------------- #
# (8) /v1/stats: facts/facts_current come from the graph.
# --------------------------------------------------------------------------- #
def test_stats_facts_from_graph(client, make_tenant):
    graph.configure_clients(**_clients())
    t = make_tenant()
    engine = _engine_for(t)
    _ada, closed, open_ = _seed_ada_moves(engine)  # 2 edges, 1 closed / 1 open

    resp = client.get("/v1/stats", headers={"X-API-Key": t.api_key})
    assert resp.status_code == 200
    body = resp.json()
    assert body["facts"] == 2
    assert body["facts_current"] == 1
    # Postgres-sourced counts are still present.
    assert set(body) == {"episodes", "entities", "facts", "facts_current"}


# --------------------------------------------------------------------------- #
# (9) /v1/graph endpoint envelope (full wiring).
# --------------------------------------------------------------------------- #
def test_graph_endpoint_envelope(client, make_tenant):
    graph.configure_clients(**_clients())
    t = make_tenant()
    engine = _engine_for(t)
    _seed_ada_moves(engine)

    resp = client.get("/v1/graph", headers={"X-API-Key": t.api_key})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"as_of", "at", "include_historical", "nodes", "edges",
                         "counts"}
    assert body["as_of"] is None and body["at"] is not None
    assert set(body["counts"]) == {
        "nodes", "edges", "entities", "episodes", "episodes_total", "sources"
    }
    # Default (as-of now) shows only the open window.
    assert body["counts"]["edges"] == 1


def test_graph_episode_layer_reflects_ledger_provenance(
        client, make_tenant, tenant_conn):
    t = make_tenant()
    headers = {"X-API-Key": t.api_key}
    with tenant_conn(t.tenant_id) as conn:
        episode_id = conn.execute(text(
            "INSERT INTO episodes (tenant_id, source, source_ref, occurred_at,"
            " content, content_hash, meta) VALUES (:t, 'manual', 'source-1',"
            " '2025-01-15T12:00:00Z', :content, :hash, '{}'::jsonb)"
            " RETURNING episode_id"
        ), {"t": str(t.tenant_id), "content": "BluePeak approved the graph.",
            "hash": uuid4().hex}).scalar_one()

    visible = client.get(
        "/v1/graph",
        headers=headers,
        params={"include_episodes": "true"},
    )
    assert visible.status_code == 200, visible.text
    body = visible.json()
    assert body["counts"] == {
        "nodes": 2, "edges": 1, "entities": 0,
        "episodes": 1, "episodes_total": 1, "sources": 1,
    }
    episode = next(n for n in body["nodes"] if n["node_kind"] == "episode")
    assert episode["id"] == f"episode:{episode_id}"
    assert episode["content"] == "BluePeak approved the graph."
    assert episode["source"] == "manual"
    assert episode["source_ref"] == "source-1"

    before = client.get(
        "/v1/graph",
        headers=headers,
        params={
            "include_episodes": "true",
            "as_of": "2025-01-01T00:00:00Z",
        },
    ).json()
    assert before["counts"]["episodes"] == 0
