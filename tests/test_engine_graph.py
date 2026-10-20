"""Lifecycle + fulltext smoke tests for app/engine/graph.py (FalkorDB engine).

These tests are also the design's smoke proof: they assert that BM25 is ALIVE
(the fulltext indices built by ``build_indices_and_constraints`` return rows),
that recipe deep-copy defeats the shared-mutation footgun, and that the
lifecycle (cache identity, idempotent close, wipe via GRAPH.DELETE, reopen
after ``close_all``) behaves against the embedded falkordblite server.

No LLM is ever called: the fake LLM client's ``_generate_response`` raises, so a
lifecycle path that accidentally reaches the model fails loudly. The embedder is
deterministic (hash-seeded 384-dim vectors) — real vectors, no network.
"""
from __future__ import annotations

# Import-ordering contract: app.engine (which sets EMBEDDING_DIM=384) must be
# imported before graphiti_core anywhere. Importing app.engine.graph pulls the
# package __init__ first; we still name app.engine explicitly here so a reader
# sees the dependency and a linter cannot reorder it below graphiti_core.
import app.engine  # noqa: F401
import app.engine.graph as graph

import hashlib
import struct
import sys

import pytest

from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.llm_client.client import LLMClient


# --- fakes ------------------------------------------------------------------


class StubLLMClient(LLMClient):
    """An LLM client that must never be invoked. Any call is a test bug — the
    lifecycle/fulltext paths are LLM-free by construction."""

    def __init__(self):
        super().__init__(config=None)

    async def _generate_response(self, messages, response_model=None, max_tokens=None, model_size=None):  # noqa: D401
        raise AssertionError("lifecycle tests must not call the LLM")


class FakeEmbedder(EmbedderClient):
    """Deterministic 384-dim embeddings, hash-seeded from the input text. No
    network, stable across runs — enough for fulltext/lifecycle tests, which
    never inspect embedding quality."""

    DIM = 384

    def _vec(self, text: str) -> list[float]:
        out: list[float] = []
        counter = 0
        while len(out) < self.DIM:
            digest = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
            # 8 floats per 32-byte digest.
            for i in range(0, 32, 4):
                (u,) = struct.unpack("<I", digest[i : i + 4])
                out.append((u / 0xFFFFFFFF) * 2.0 - 1.0)
            counter += 1
        return out[: self.DIM]

    async def create(self, input_data):
        if isinstance(input_data, str):
            return self._vec(input_data)
        # graphiti passes list[str]/token sequences; join to a stable key.
        return self._vec(repr(input_data))

    async def create_batch(self, input_data_list):
        return [self._vec(t) for t in input_data_list]


class FakeCrossEncoder(CrossEncoderClient):
    """Passthrough reranker (identity order, descending score). The RRF /
    node-distance recipes never call it, but Graphiti requires one."""

    async def rank(self, query, passages):
        n = len(passages)
        return [(p, float(n - i)) for i, p in enumerate(passages)]


def _clients():
    return dict(
        llm_client=StubLLMClient(),
        embedder=FakeEmbedder(),
        cross_encoder=FakeCrossEncoder(),
    )


@pytest.fixture(autouse=True)
def _sandbox_graph_dir(tmp_path, monkeypatch):
    """Point MEMORY_GRAPH_DIR at a per-test tmp dir (its own embedded server)
    and guarantee every cached engine AND the embedded falkordblite server are
    torn down afterward — no redis-server child may outlive the test."""
    monkeypatch.setenv("MEMORY_GRAPH_DIR", str(tmp_path / "graphdb"))
    yield
    graph.close_all()


# --- engine creation + schema ----------------------------------------------


def _query(engine, cypher, **params):
    async def _go():
        rows, _, _ = await engine.driver.execute_query(cypher, **params)
        return rows

    # graph._run_coro, NOT asyncio.run: the async FalkorDB client is bound to
    # the persistent graph loop; a fresh per-call loop would corrupt its pool.
    return graph._run_coro(_go())


def test_engine_builds_indices():
    engine = graph.get_engine("tenantA", **_clients())
    # build_indices_and_constraints ran at engine creation: fulltext + range
    # indices exist for the Graphiti labels (db.indexes lists them).
    rows = _query(engine, "CALL db.indexes() YIELD label RETURN label")
    labels = {r["label"] for r in rows}
    assert {"Entity", "Episodic", "RELATES_TO"} <= labels, labels


def test_cache_identity_same_tenant():
    a = graph.get_engine("tenantA", **_clients())
    b = graph.get_engine("tenantA", **_clients())
    assert a is b


def test_cache_distinct_on_changed_root(tmp_path, monkeypatch):
    a = graph.get_engine("tenantA", **_clients())
    # A changed MEMORY_GRAPH_DIR must yield a distinct instance, not a stale one.
    monkeypatch.setenv("MEMORY_GRAPH_DIR", str(tmp_path / "graphdb2"))
    b = graph.get_engine("tenantA", **_clients())
    assert a is not b


def test_close_is_idempotent():
    engine = graph.get_engine("tenantA", **_clients())
    engine.close()
    engine.close()  # must not raise
    # A closed engine refuses to hand out native handles.
    with pytest.raises(RuntimeError):
        _ = engine.driver


# --- fulltext smoke: BM25 is alive -------------------------------------------


def test_fulltext_indices_return_rows():
    """Insert an Entity node and a native RELATES_TO relationship with
    distinctive text, then prove the fulltext procedures return them for both
    the node and relationship indices — i.e. BM25 is wired, not silently dead."""
    engine = graph.get_engine("tenantFTS", **_clients())

    _query(
        engine,
        "CREATE (:Entity {uuid:'e1', name:'Xylophone Wizard', "
        "group_id:'tenantFTS', summary:'plays the marimba at dawn'})",
    )
    _query(
        engine,
        "CREATE (:Entity {uuid:'e2', name:'Marimba Guild', group_id:'tenantFTS'})",
    )
    _query(
        engine,
        "MATCH (a:Entity {uuid:'e1'}), (b:Entity {uuid:'e2'}) "
        "CREATE (a)-[:RELATES_TO {uuid:'r1', group_id:'tenantFTS', "
        "name:'MEMBER_OF', "
        "fact:'Xylophone Wizard is a member of the Marimba Guild'}]->(b)",
    )

    node_hits = _query(
        engine,
        "CALL db.idx.fulltext.queryNodes('Entity', 'marimba') YIELD node "
        "RETURN node.uuid AS uuid",
    )
    assert any(r["uuid"] == "e1" for r in node_hits), node_hits

    edge_hits = _query(
        engine,
        "CALL db.idx.fulltext.queryRelationships('RELATES_TO', 'Guild') "
        "YIELD relationship RETURN relationship.uuid AS uuid",
    )
    assert any(r["uuid"] == "r1" for r in edge_hits), edge_hits


def test_reopen_existing_graph_after_close_all():
    """Second get_engine on the same root (after close_all shut the embedded
    server down) must reattach to the persisted data without failing on the
    already-existing indices, and fulltext must still work."""
    engine = graph.get_engine("tenantReopen", **_clients())
    _query(
        engine,
        "CREATE (:Entity {uuid:'e1', name:'Quokka Diplomat', "
        "group_id:'tenantReopen', summary:'negotiates treaties'})",
    )
    graph.close_all()  # stops the embedded server; data persists on disk

    reopened = graph.get_engine("tenantReopen", **_clients())
    assert reopened is not engine  # genuinely reconstructed
    hits = _query(
        reopened,
        "CALL db.idx.fulltext.queryNodes('Entity', 'treaties') YIELD node "
        "RETURN node.uuid AS uuid",
    )
    assert any(r["uuid"] == "e1" for r in hits), hits


# --- recipe deep-copy guard -------------------------------------------------


def test_recipe_returns_distinct_objects():
    a = graph.recipe("EDGE_HYBRID_SEARCH_RRF")
    b = graph.recipe("EDGE_HYBRID_SEARCH_RRF")
    assert a is not b
    assert a.edge_config is not b.edge_config


def test_recipe_mutation_does_not_leak():
    a = graph.recipe("EDGE_HYBRID_SEARCH_RRF")
    a.limit = 999  # simulate Graphiti.search() mutating the shared object
    b = graph.recipe("EDGE_HYBRID_SEARCH_RRF")
    assert b.limit != 999
    # And the shared source object is untouched.
    from graphiti_core.search import search_config_recipes as R

    assert R.EDGE_HYBRID_SEARCH_RRF.limit != 999


def test_recipe_limit_survives_deep_copy():
    r = graph.recipe("EDGE_HYBRID_SEARCH_RRF")
    r.limit = 7
    assert r.limit == 7  # deep copy preserves an assignable, independent limit


def test_recipe_unknown_name_raises():
    with pytest.raises(ValueError):
        graph.recipe("NOPE_NOT_A_RECIPE")


def test_all_required_recipes_supported():
    for name in (
        "EDGE_HYBRID_SEARCH_RRF",
        "NODE_HYBRID_SEARCH_RRF",
        "EDGE_HYBRID_SEARCH_NODE_DISTANCE",
        "COMBINED_HYBRID_SEARCH_RRF",
    ):
        assert isinstance(graph.recipe(name), type(graph.recipe(name)))


# --- wipe / existence ---------------------------------------------------------


def test_wipe_removes_graph_and_cache():
    engine = graph.get_engine("tenantWipe", **_clients())
    _query(engine, "CREATE (:Entity {uuid:'w1', name:'Doomed', "
                   "group_id:'tenantWipe'})")
    assert graph.graph_exists("tenantWipe")

    graph.wipe("tenantWipe")
    assert not graph.graph_exists("tenantWipe")

    # Dropped from cache: a new get_engine builds a fresh instance.
    fresh = graph.get_engine("tenantWipe", **_clients())
    assert fresh is not engine
    rows = _query(fresh, "MATCH (n:Entity) RETURN count(n) AS c")
    assert rows[0]["c"] == 0  # genuinely empty, not the old graph


def test_wipe_absent_tenant_is_noop():
    graph.wipe("never-existed")  # must not raise


def test_configure_clients_supplies_defaults():
    """get_engine() with no explicit clients uses the process-wide set registered
    via configure_clients(); an unconfigured process raises a clear error."""
    saved = graph._clients
    graph._clients = None
    try:
        with pytest.raises(RuntimeError):
            graph.get_engine("tenantNoClients")
        graph.configure_clients(**_clients())
        engine = graph.get_engine("tenantConfigured")
        assert engine.graphiti is not None
    finally:
        graph._clients = saved


# --- full Graphiti construction + empty search (LLM-free) -------------------


def test_graphiti_constructs_and_empty_search_skips_llm():
    """Graphiti(graph_driver, llm_client=stub, embedder=fake, cross_encoder=fake)
    constructs, and a search() over an empty graph returns [] without touching
    the LLM stub (which would raise)."""
    engine = graph.get_engine("tenantSearch", **_clients())
    g = engine.graphiti

    async def _search():
        # Deep-copied recipe so the shared object is never mutated.
        return await g.search(
            "anything at all",
            group_ids=[engine.group_id],
            num_results=5,
        )

    edges = graph._run_coro(_search())
    assert edges == []


# --- health + import ordering -------------------------------------------------


def test_health_check_reaches_the_deployment():
    graph.health_check()  # boots the embedded server if needed; raises on failure


def test_import_ordering_app_engine_before_graphiti():
    """app.engine must be imported before graphiti_core so EMBEDDING_DIM=384 is
    set before graphiti reads it as a module-level constant."""
    assert "app.engine" in sys.modules
    assert "graphiti_core" in sys.modules
    # The dimension graphiti will pad zero-vectors to matches our embedder.
    from graphiti_core.embedder.client import EMBEDDING_DIM

    assert EMBEDDING_DIM == 384
