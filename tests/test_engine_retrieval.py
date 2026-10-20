"""Tests for app/engine/retrieval.py — entity-centric /v1/context assembly.

Graph state is built DIRECTLY (EntityNode.save / EntityEdge.save on the tenant
engine's driver) — no LLM anywhere: the stub LLM client raises on any call, so a
retrieval path that accidentally reaches the model fails loudly. Embeddings are
REAL (FastEmbedEmbedder, bge-small-en-v1.5, local ONNX — offline) so hybrid
search actually ranks. Postgres is used only for what retrieval writes there:
the ``retrieval.route`` audit row.
"""
from __future__ import annotations

# Import-ordering contract: app.engine sets EMBEDDING_DIM=384 before any
# graphiti_core import (see app/engine/__init__.py).
import app.engine  # noqa: F401
import app.engine.graph as graph

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import text

from graphiti_core.edges import EntityEdge
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType

from app.engine.clients import FastEmbedEmbedder, PassthroughCrossEncoder
from app.engine.retrieval import (
    _query_coverage,
    _rank_raw_passages_async,
    _search_async,
    assemble_context,
    estimate_tokens,
    extract_mentions,
    native_search_filters,
)
from app.episodes import ingest_episode

_UTC = timezone.utc


def D(y, m, d):
    return datetime(y, m, d, tzinfo=_UTC)


class StubLLMClient(LLMClient):
    """Must never be invoked: retrieval is LLM-free by construction."""

    def __init__(self):
        super().__init__(config=None)

    async def _generate_response(self, messages, response_model=None,
                                 max_tokens=None, model_size=None):
        raise AssertionError("retrieval must not call the LLM")


# One real embedder for the whole module (the ONNX model loads once).
EMBEDDER = FastEmbedEmbedder()


def _clients():
    return dict(llm_client=StubLLMClient(), embedder=EMBEDDER,
                cross_encoder=PassthroughCrossEncoder())


@pytest.fixture(autouse=True)
def _sandbox_graph_dir(tmp_path, monkeypatch):
    """Per-test MEMORY_GRAPH_DIR + deterministic native teardown."""
    monkeypatch.setenv("MEMORY_GRAPH_DIR", str(tmp_path / "graphdb"))
    yield
    graph.close_all()


# --------------------------------------------------------------------------- #
# Graph seeding helpers (direct save, no LLM).
# --------------------------------------------------------------------------- #
def _node(engine, name, *, summary="", labels=("Person",)):
    return EntityNode(uuid=str(uuid4()), name=name, group_id=engine.group_id,
                      labels=list(labels), summary=summary)


def _edge(engine, src, tgt, name, fact, *, valid_at=None, invalid_at=None,
          episodes=("ep1",)):
    return EntityEdge(
        uuid=str(uuid4()), source_node_uuid=src.uuid, target_node_uuid=tgt.uuid,
        name=name, fact=fact, group_id=engine.group_id,
        created_at=D(2024, 1, 1), valid_at=valid_at, invalid_at=invalid_at,
        episodes=list(episodes))


def _seed(engine, nodes, edges, episodes=()):
    async def _go():
        for n in nodes:
            await n.generate_name_embedding(EMBEDDER)
            await n.save(engine.driver)
        for e in edges:
            await e.generate_embedding(EMBEDDER)
            await e.save(engine.driver)
        for ep in episodes:
            await ep.save(engine.driver)  # episode search is BM25-only, no embed

    graph._run_coro(_go())


def _episode(engine, content, *, valid_at=None, source_description="chat"):
    """A raw EpisodicNode (no LLM, no entity extraction) so hybrid search's
    BM25-only episode leg has something to return."""
    return EpisodicNode(
        uuid=str(uuid4()), name="session", group_id=engine.group_id,
        source=EpisodeType.message, source_description=source_description,
        content=content, valid_at=valid_at or D(2024, 1, 1), entity_edges=[])


def _engine_for(tenant):
    return graph.get_engine(str(tenant.tenant_id), **_clients())


def _ctx(conn, tenant, query, **kw):
    return assemble_context(conn, tenant.tenant_id, query=query, **kw)


# --------------------------------------------------------------------------- #
# (1) Fan-out: ALL of an entity's edges come back — the FM-1 proof.
# --------------------------------------------------------------------------- #
def test_fanout_returns_all_entity_edges(make_tenant, tenant_conn):
    t = make_tenant()
    engine = _engine_for(t)
    marisol = _node(engine, "Marisol Quintero", summary="a cellist based in Porto")
    targets = [
        ("moved_to", "Marisol Quintero moved to Porto", "Porto"),
        ("adopted", "Marisol Quintero adopted a greyhound named Faro", "Faro"),
        ("works_at", "Marisol Quintero works at the harbor observatory", "Harbor Observatory"),
        ("plays", "Marisol Quintero plays the cello", "Cello"),
        ("allergic_to", "Marisol Quintero is allergic to peanuts", "Peanuts"),
        ("studied_at", "Marisol Quintero studied at Coimbra", "Coimbra"),
    ]
    nodes = [marisol]
    edges = []
    for i, (rel, fact, tname) in enumerate(targets):
        tgt = _node(engine, tname, labels=("Thing",))
        nodes.append(tgt)
        edges.append(_edge(engine, marisol, tgt, rel, fact,
                           valid_at=D(2024, 1, 1 + i)))
    _seed(engine, nodes, edges)

    with tenant_conn(t.tenant_id) as conn:
        resp = _ctx(conn, t, "Tell me everything about Marisol Quintero",
                    fmt="json")

    got_uuids = {it.get("uuid") for it in resp["items"]}
    for e in edges:
        assert e.uuid in got_uuids, f"fan-out missed edge {e.name}: {e.fact}"
    assert resp["tokens_estimated"] <= resp["budget"]


def test_fanout_includes_object_side_edges(make_tenant, tenant_conn):
    """An edge where the queried entity is the TARGET (object side) must be
    retrieved too — invisible in the old engine."""
    t = make_tenant()
    engine = _engine_for(t)
    ada = _node(engine, "Ada Vermeer")
    club = _node(engine, "Harbor Chess Club", labels=("Organization",))
    e = _edge(engine, club, ada, "founded_by",
              "The Harbor Chess Club was founded by Ada Vermeer",
              valid_at=D(2024, 3, 1))
    _seed(engine, [ada, club], [e])

    with tenant_conn(t.tenant_id) as conn:
        resp = _ctx(conn, t, "Who is Ada Vermeer?", fmt="json")
    assert any(it.get("uuid") == e.uuid for it in resp["items"])


def test_falkordb_native_bfs_expands_from_target_side(make_tenant):
    """The FalkorDB adapter must match Graphiti's endpoint-neutral BFS semantics."""
    t = make_tenant()
    engine = _engine_for(t)
    ada = _node(engine, "Ada Vermeer")
    club = _node(engine, "Harbor Chess Club", labels=("Organization",))
    porto = _node(engine, "Porto", labels=("Place",))
    target_side = _edge(
        engine, club, ada, "founded_by", "The club was founded by Ada Vermeer")
    second_hop = _edge(
        engine, club, porto, "based_in", "The Harbor Chess Club is based in Porto")
    _seed(engine, [ada, club, porto], [target_side, second_hop])

    result = graph._run_coro(_search_async(
        engine,
        "unrelated seed",
        limit=10,
        filters=native_search_filters(
            as_of=None, as_believed_at=None, include_historical=False),
        origins=[ada.uuid],
    ))
    assert {target_side.uuid, second_hop.uuid} <= {edge.uuid for edge in result.edges}


# --------------------------------------------------------------------------- #
# (2) as_of excludes a closed window; include_historical keeps it.
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
    return closed, open_


def test_as_of_excludes_closed_window(make_tenant, tenant_conn):
    t = make_tenant()
    engine = _engine_for(t)
    closed, open_ = _seed_ada_moves(engine)

    with tenant_conn(t.tenant_id) as conn:
        after = _ctx(conn, t, "Where does Ada Vermeer live?", fmt="json",
                     as_of=D(2025, 1, 1))
        before = _ctx(conn, t, "Where does Ada Vermeer live?", fmt="json",
                      as_of=D(2023, 6, 1))

    after_uuids = {it.get("uuid") for it in after["items"]}
    assert open_.uuid in after_uuids
    assert closed.uuid not in after_uuids

    before_uuids = {it.get("uuid") for it in before["items"]}
    assert closed.uuid in before_uuids
    assert open_.uuid not in before_uuids


def test_include_historical_keeps_closed_window(make_tenant, tenant_conn):
    t = make_tenant()
    engine = _engine_for(t)
    closed, open_ = _seed_ada_moves(engine)

    with tenant_conn(t.tenant_id) as conn:
        default = _ctx(conn, t, "Where does Ada Vermeer live?", fmt="json")
        hist = _ctx(conn, t, "Where does Ada Vermeer live?", fmt="json",
                    include_historical=True)

    assert closed.uuid not in {it.get("uuid") for it in default["items"]}
    hist_uuids = {it.get("uuid") for it in hist["items"]}
    assert {closed.uuid, open_.uuid} <= hist_uuids


# Graphiti may keep multiple open values; retrieval must not second-guess them.
def test_no_application_older_value_annotation(make_tenant, tenant_conn):
    t = make_tenant()
    engine = _engine_for(t)
    nadia = _node(engine, "Nadia Rooke")
    blue = _node(engine, "Blue", labels=("Thing",))
    green = _node(engine, "Green", labels=("Thing",))
    older = _edge(engine, nadia, blue, "favorite_color",
                  "Nadia Rooke's favorite color is blue", valid_at=D(2024, 1, 1))
    newer = _edge(engine, nadia, green, "favorite_color",
                  "Nadia Rooke's favorite color is green", valid_at=D(2024, 6, 1))
    timeless = _edge(engine, nadia, blue, "likes",
                     "Nadia Rooke likes the color blue")  # no temporal claim
    _seed(engine, [nadia, blue, green], [older, newer, timeless])

    with tenant_conn(t.tenant_id) as conn:
        resp = _ctx(conn, t, "What is Nadia Rooke's favorite color?", fmt="json")

    by_uuid = {it.get("uuid"): it for it in resp["items"] if "uuid" in it}
    assert by_uuid[older.uuid]["older_value"] is False
    assert "[older value]" not in by_uuid[older.uuid]["line"]
    assert by_uuid[newer.uuid]["older_value"] is False
    assert "[older value]" not in by_uuid[newer.uuid]["line"]
    # Regression: a timeless edge makes no temporal claim — never marked older.
    assert by_uuid[timeless.uuid]["older_value"] is False


# --------------------------------------------------------------------------- #
# Native dated evidence is rendered without synthetic date arithmetic.
# --------------------------------------------------------------------------- #
def test_no_synthetic_delta_lines(make_tenant, tenant_conn):
    t = make_tenant()
    engine = _engine_for(t)
    piet = _node(engine, "Piet Hein")
    uku = _node(engine, "Ukulele Lessons", labels=("Thing",))
    gig = _node(engine, "Harbor Concert", labels=("Thing",))
    a = _edge(engine, piet, uku, "started",
              "Piet Hein started ukulele lessons", valid_at=D(2024, 1, 1))
    b = _edge(engine, piet, gig, "performed_at",
              "Piet Hein performed at the harbor concert", valid_at=D(2024, 1, 25))
    _seed(engine, [piet, uku, gig], [a, b])

    with tenant_conn(t.tenant_id) as conn:
        resp = _ctx(conn, t, "Piet Hein ukulele and concert timeline")

    assert "Δ" not in resp["block"]
    assert "2024-01-01" in resp["block"]
    assert "2024-01-25" in resp["block"]


# --------------------------------------------------------------------------- #
# (6) Abstention: a named entity that does not exist → empty context, but ONLY
# when the hybrid search ALSO surfaced nothing (no edges, no episodes).
# --------------------------------------------------------------------------- #
def test_abstention_on_nonexistent_entity(make_tenant, tenant_conn):
    """A named entity resolves to no node AND the search returns zero edges and
    zero episodes → true false premise → empty context. (Zorblatt is unrelated to
    the Ada graph, so hybrid search — with sim_min_score=0.6 — surfaces nothing.)"""
    t = make_tenant()
    engine = _engine_for(t)
    _seed_ada_moves(engine)  # a POPULATED graph

    with tenant_conn(t.tenant_id) as conn:
        resp = _ctx(conn, t, "What does Zorblatt Enterprises manufacture?")

    assert resp["items_returned"] == 0
    assert resp["block"] == ""


def test_no_abstention_when_search_finds_episodes(make_tenant, tenant_conn):
    """A query naming an entity that resolves to NO node must still return the
    episode search hits when the hybrid search found material — the mention is a
    value we never node-ify (an airline). Regression for the over-firing guard
    that emptied the American Airlines / United questions."""
    t = make_tenant()
    engine = _engine_for(t)
    ep = _episode(
        engine,
        "I flew Zephyr Airlines to Boston on Valentine's day for the offsite.",
        valid_at=D(2024, 2, 14))
    _seed(engine, [], [], episodes=[ep])

    with tenant_conn(t.tenant_id) as conn:
        resp = _ctx(conn, t, "What airline did I fly with on Valentine's day?",
                    fmt="json")

    # Named ("Valentine's") but unresolved — NOT abstained: the episode carries it.
    assert resp["items_returned"] > 0
    assert any(it.get("episode_id") == str(ep.uuid) for it in resp["items"])
    assert any("Zephyr Airlines" in (it.get("snippet") or "")
               for it in resp["items"])


def test_episode_blended_even_with_many_edges(make_tenant, tenant_conn):
    """The top episode hit is blended on a default request even when ≥5 edges
    already resolved (single-session detail that never became an edge). Regression
    for the BACKFILL_MIN_EDGES gate suppressing episodes on rich entities."""
    t = make_tenant()
    engine = _engine_for(t)
    rafael = _node(engine, "Rafael Ortiz", summary="an archivist in Lisbon")
    nodes = [rafael]
    edges = []
    for i in range(6):  # ≥5 edges → the old backfill gate would add NO episodes
        tgt = _node(engine, f"Topic{i}", labels=("Thing",))
        nodes.append(tgt)
        edges.append(_edge(engine, rafael, tgt, f"rel_{i}",
                           f"Rafael Ortiz is associated with topic number {i}",
                           valid_at=D(2024, 1, 1 + i)))
    ep = _episode(
        engine,
        "Rafael Ortiz mentioned his family heritage is an Irish and Italian mix.",
        valid_at=D(2024, 3, 1))
    _seed(engine, nodes, edges, episodes=[ep])

    with tenant_conn(t.tenant_id) as conn:
        resp = _ctx(conn, t, "Tell me everything about Rafael Ortiz", fmt="json")

    # ≥5 edge items resolved, yet the episode is still blended in.
    edge_items = [it for it in resp["items"] if "uuid" in it]
    assert len(edge_items) >= 5
    assert any(it.get("episode_id") == str(ep.uuid) for it in resp["items"])


def test_raw_passages_preserve_multiple_support_sessions(make_tenant, tenant_conn):
    """Source diversity prevents one transcript from consuming the whole budget."""
    t = make_tenant()
    _engine_for(t)
    with tenant_conn(t.tenant_id) as conn:
        first, _ = ingest_episode(
            conn, t.tenant_id, source="chat", occurred_at=D(2024, 2, 1),
            content="User: I attended a two-day pottery workshop last weekend.")
        second, _ = ingest_episode(
            conn, t.tenant_id, source="chat", occurred_at=D(2024, 3, 1),
            content="User: I also went to a one-day library lecture in March.")

    with tenant_conn(t.tenant_id) as conn:
        resp = _ctx(
            conn, t,
            "How many days of workshops and lectures did I attend in total?",
            fmt="json", include="knowledge")

    episode_ids = {item.get("episode_id") for item in resp["items"]}
    assert {str(first), str(second)} <= episode_ids
    rendered = " ".join(item.get("snippet", "") for item in resp["items"])
    assert "two-day pottery workshop" in rendered
    assert "one-day library lecture" in rendered


def test_raw_passage_selects_relevant_late_turn_not_transcript_prefix(
        make_tenant, tenant_conn):
    t = make_tenant()
    _engine_for(t)
    transcript = (
        "Assistant: " + "Generic fishing advice. " * 100 + "\n"
        "User: At the derby on Saturday I caught exactly 12 bass in Lake Michigan."
    )
    with tenant_conn(t.tenant_id) as conn:
        ingest_episode(conn, t.tenant_id, source="chat", occurred_at=D(2024, 4, 1),
                       content=transcript)
    with tenant_conn(t.tenant_id) as conn:
        resp = _ctx(conn, t, "How many bass did I catch at the derby?", fmt="json")
    assert any("exactly 12 bass" in (item.get("snippet") or "")
               for item in resp["items"])


def test_raw_passage_lexical_lane_survives_same_entity_distractors():
    """Repeated entity mentions cannot crowd an exact answer out of the pool.

    The embedder deliberately makes 100 generic Cedar Labs entries perfect
    dense matches and the answer orthogonal.  The answer still has to win the
    high-confidence lexical lane through ``expansion/expand`` and
    ``seat/seats`` inflection matching.
    """

    class DenseTrapEmbedder:
        async def create_batch(self, texts):
            return [
                [0.0, 1.0] if "EU data residency" in item else [1.0, 0.0]
                for item in texts
            ]

    question = "What blocks Cedar Labs' seat expansion?"
    rows = [
        SimpleNamespace(
            episode_id=f"distractor-{index}",
            source="email",
            occurred_at=D(2026, 6, 1),
            content=(
                f"Product feedback {index}: Cedar Labs requested saved search "
                "alerts at high priority."
            ),
            meta={},
        )
        for index in range(100)
    ]
    rows.append(SimpleNamespace(
        episode_id="answer",
        source="email",
        occurred_at=D(2026, 6, 2),
        content=(
            "Cedar Labs requires EU data residency before it will expand to "
            "500 seats."
        ),
        meta={},
    ))
    engine = SimpleNamespace(graphiti=SimpleNamespace(
        embedder=DenseTrapEmbedder(),
        cross_encoder=PassthroughCrossEncoder(),
    ))

    ranked = graph._run_coro(_rank_raw_passages_async(
        engine, [question, "Cedar Labs"], rows, set()
    ))

    assert _query_coverage(question, rows[-1].content) == pytest.approx(0.8)
    assert _query_coverage(question, rows[0].content) == pytest.approx(0.4)
    assert ranked[0].episode_id == "answer"


def test_current_query_prefers_newer_raw_source(make_tenant, tenant_conn):
    """Explicit current-state questions rank the latest source assertion first."""
    t = make_tenant()
    _engine_for(t)
    with tenant_conn(t.tenant_id) as conn:
        ingest_episode(
            conn, t.tenant_id, source="chat", occurred_at=D(2024, 1, 1),
            content="User: I currently keep my old sneakers under the bed.")
        newer, _ = ingest_episode(
            conn, t.tenant_id, source="chat", occurred_at=D(2024, 2, 1),
            content="User: I currently keep my old sneakers in the closet shoe rack.")

    with tenant_conn(t.tenant_id) as conn:
        resp = _ctx(
            conn, t, "Where do I currently keep my old sneakers?",
            fmt="json", include="knowledge",
        )

    sources = [item for item in resp["items"] if item.get("episode_id")]
    assert sources
    assert sources[0]["episode_id"] == str(newer)
    assert "closet shoe rack" in sources[0]["snippet"]


def test_no_abstention_without_named_entities(make_tenant, tenant_conn):
    """An all-lowercase query has no capitalized mentions — never abstain."""
    t = make_tenant()
    engine = _engine_for(t)
    _seed_ada_moves(engine)

    with tenant_conn(t.tenant_id) as conn:
        resp = _ctx(conn, t, "who lives in lisbon these days?", fmt="json")
    assert resp["items_returned"] > 0


def test_extract_mentions_conservative():
    # Sentence-initial question words are not mentions.
    assert extract_mentions("Where does she live?") == []
    # Quoted spans are mentions even when lowercase.
    assert extract_mentions('what is "project nimbus"?') == ["project nimbus"]
    # Capitalized runs are mentions, leading stopword stripped.
    assert extract_mentions("What Zorblatt Enterprises makes") == [
        "Zorblatt Enterprises"]


# --------------------------------------------------------------------------- #
# (7) Budget: the block never exceeds tiny budgets.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("budget", [25, 60])
def test_budget_hard_cap(make_tenant, tenant_conn, budget):
    t = make_tenant()
    engine = _engine_for(t)
    _seed_ada_moves(engine)

    with tenant_conn(t.tenant_id) as conn:
        resp = _ctx(conn, t, "Where does Ada Vermeer live?",
                    token_budget=budget)

    assert resp["tokens_estimated"] <= budget
    assert estimate_tokens(resp["block"]) <= budget
    if budget >= 60:
        # A single edge line always fits 60 tokens; 25 may fit none (uuid
        # prefixes tokenize to varying counts) — zero items is then correct.
        assert resp["items_returned"] >= 1


# --------------------------------------------------------------------------- #
# (8) Response contract + the retrieval.route audit row.
# --------------------------------------------------------------------------- #
def test_response_contract_and_audit_row(make_tenant, tenant_conn):
    t = make_tenant()
    engine = _engine_for(t)
    _seed_ada_moves(engine)

    with tenant_conn(t.tenant_id) as conn:
        resp = _ctx(conn, t, "Where does Ada Vermeer live?")
        rows = conn.execute(
            text("SELECT detail FROM audit_log WHERE tenant_id = :t "
                 "AND action = 'retrieval.route'"),
            {"t": str(t.tenant_id)}).mappings().all()

    assert set(resp) == {"route", "items_returned", "tokens_estimated",
                         "budget", "stratum_mix", "cache_stable_hash", "block"}
    assert resp["route"] in ("facts", "knowledge", "merged")
    assert resp["cache_stable_hash"] is None
    assert set(resp["stratum_mix"]) == {"temporal", "semantic"}
    # Dated edges count as temporal stratum items.
    assert resp["stratum_mix"]["temporal"] >= 1

    assert len(rows) == 1
    detail = rows[0]["detail"]
    # /v1/admin/token_stats reads route, tokens, stratum_mix from this row.
    assert detail["route"] == resp["route"]
    assert detail["tokens"] == resp["tokens_estimated"]
    assert detail["budget"] == resp["budget"]
    assert detail["stratum_mix"] == resp["stratum_mix"]
    assert detail["query"] == "Where does Ada Vermeer live?"
    assert detail["overridden"] is False
    assert detail["cache_stable_hash"] is None


def test_native_belief_time_filter_and_ignored_dispute_flag(make_tenant, tenant_conn):
    t = make_tenant()
    engine = _engine_for(t)
    _seed_ada_moves(engine)

    with tenant_conn(t.tenant_id) as conn:
        plain = _ctx(conn, t, "Where does Ada Vermeer live?")
        before_creation = _ctx(conn, t, "Where does Ada Vermeer live?",
                               as_believed_at=D(2023, 1, 1))
        with_disputed = _ctx(conn, t, "Where does Ada Vermeer live?",
                             include_disputed=True)
    assert before_creation["block"] == ""
    assert with_disputed["block"] == plain["block"]


# --------------------------------------------------------------------------- #
# (9) ABAC: out-of-scope edges are dropped.
# --------------------------------------------------------------------------- #
def test_abac_scope_filtering(make_tenant, tenant_conn):
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
        resp = _ctx(conn, t, "Tell me about Rosa Marin and Tomas Berg",
                    fmt="json", allowed_subject_scopes=[rosa.uuid])

    uuids = {it.get("uuid") for it in resp["items"]}
    assert in_scope.uuid in uuids
    assert out_scope.uuid not in uuids


# --------------------------------------------------------------------------- #
# Thin /v1/context endpoint test (TestClient): the wiring, not the pipeline.
# --------------------------------------------------------------------------- #
def test_context_endpoint(client, make_tenant):
    graph.configure_clients(**_clients())  # what process startup would do
    t = make_tenant()
    engine = _engine_for(t)
    _seed_ada_moves(engine)
    hdr = {"X-API-Key": t.api_key}

    resp = client.post("/v1/context", headers=hdr,
                       json={"query": "Where does Ada Vermeer live?",
                             "tier_floor": 2})  # tier_floor accepted + ignored
    assert resp.status_code == 200
    body = resp.json()
    assert "Lisbon" in body["block"]
    assert body["tokens_estimated"] <= body["budget"]
    assert body["cache_stable_hash"] is None

    # Legacy interval operators are rejected with a migration message.
    bad = client.post("/v1/context", headers=hdr,
                      json={"query": "x", "interval_op": "sometimes",
                            "ref_interval": {"start": "2024-01-01T00:00:00Z"}})
    assert bad.status_code == 422
    missing_ref = client.post("/v1/context", headers=hdr,
                              json={"query": "x", "interval_op": "overlaps"})
    assert missing_ref.status_code == 422
