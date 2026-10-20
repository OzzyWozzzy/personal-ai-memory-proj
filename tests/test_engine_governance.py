"""Governed eviction / clear / manual-fact tests for the v4 engine.

Same real pieces as tests/test_engine_ingest.py: the embedded Postgres devdb,
per-test FalkorDB graphs under tmp_path, and the REAL FastEmbedEmbedder. The only
fake is the LLM (:class:`engine_fakes.ScriptedLLMClient`), so every test is
deterministic and offline. Graph state is built through the real write path
(``ingest_text``) and, where a shared-evidence edge is needed, adjusted directly
in FalkorDB — both patterns the eviction module must handle.
"""
from __future__ import annotations

# Import-ordering contract: app.engine pins EMBEDDING_DIM=384 before any
# graphiti_core import (see app/engine/__init__.py).
import app.engine  # noqa: F401
import app.engine.graph as graph

from datetime import datetime, timezone

import pytest
from sqlalchemy import text as sql

from tests.engine_fakes import ScriptedLLMClient

from app import eviction
from app.engine.clients import FastEmbedEmbedder, PassthroughCrossEncoder
from app.engine.ingest import ingest_text

_UTC = timezone.utc
JAN = datetime(2026, 1, 5, tzinfo=_UTC)
FEB = datetime(2026, 2, 10, tzinfo=_UTC)


# --------------------------------------------------------------------------- #
# Fixtures (mirror tests/test_engine_ingest.py)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def embedder():
    return FastEmbedEmbedder()


@pytest.fixture
def graph_sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_GRAPH_DIR", str(tmp_path / "graphdb"))
    yield
    graph.close_all()


@pytest.fixture
def use_scripted(graph_sandbox, embedder):
    def _use(scripted: ScriptedLLMClient | None = None) -> ScriptedLLMClient:
        client = scripted or ScriptedLLMClient()
        graph.configure_clients(
            llm_client=client, embedder=embedder,
            cross_encoder=PassthroughCrossEncoder())
        return client

    return _use


def _cypher(tenant_id, cypher: str, **params):
    engine = graph.get_engine(str(tenant_id))

    async def _go():
        rows, _, _ = await engine.driver.execute_query(cypher, **params)
        return rows

    return graph._run_coro(_go())


def _edge_count(tenant_id) -> int:
    return _cypher(tenant_id, "MATCH ()-[e:RELATES_TO]->() RETURN count(e) AS n")[0]["n"]


def _ep_uuid(tenant_id, episode_id: str):
    rows = _cypher(tenant_id, "MATCH (e:Episodic {name: $n}) RETURN e.uuid AS uuid",
                 n=str(episode_id))
    return rows[0]["uuid"] if rows else None


def h(t):
    return {"X-API-Key": t.api_key}


# --------------------------------------------------------------------------- #
# (1) evict: exclusive edges deleted, shared-evidence edges survive
# --------------------------------------------------------------------------- #
def test_evict_removes_exclusive_edges_keeps_shared(
        make_tenant, tenant_conn, use_scripted):
    t = make_tenant()
    use_scripted(ScriptedLLMClient()
                 .add_script("Alice joined Acme", entities=["Alice", "Acme"],
                             edges=[{"source": "Alice", "target": "Acme",
                                     "relation": "WORKS_AT",
                                     "fact": "Alice works at Acme"}])
                 .add_script("Bob joined Beta", entities=["Bob", "Beta"],
                             edges=[{"source": "Bob", "target": "Beta",
                                     "relation": "WORKS_AT",
                                     "fact": "Bob works at Beta"}]))

    a = ingest_text(t.tenant_id, text="Alice joined Acme", occurred_at=JAN,
                    source="note", meta=None)
    b = ingest_text(t.tenant_id, text="Bob joined Beta", occurred_at=FEB,
                    source="note", meta=None)
    exclusive_edge = a.edges[0].uuid   # evidenced only by episode A
    shared_edge = b.edges[0].uuid      # made evidenced by A AND B below

    a_ep = _ep_uuid(t.tenant_id, a.episode_id)
    b_ep = _ep_uuid(t.tenant_id, b.episode_id)
    _cypher(t.tenant_id, "MATCH ()-[e:RELATES_TO {uuid: $u}]->() SET e.episodes = $eps",
          u=shared_edge, eps=[a_ep, b_ep])

    with tenant_conn(t.tenant_id) as conn:
        plan = eviction.evict_episode(conn, t.tenant_id, a.episode_id,
                                      actor="test")

    assert plan == {"facts": 1, "episodes": 1}

    # The exclusive edge is gone; the shared edge survives with A dropped.
    assert _cypher(t.tenant_id,
                 "MATCH ()-[e:RELATES_TO {uuid: $u}]->() RETURN e.uuid AS uuid",
                 u=exclusive_edge) == []
    surviving = _cypher(t.tenant_id,
                      "MATCH ()-[e:RELATES_TO {uuid: $u}]->() RETURN e.episodes AS eps",
                      u=shared_edge)
    assert surviving[0]["eps"] == [b_ep]

    # Episode A's graph node is gone; B's remains.
    assert _ep_uuid(t.tenant_id, a.episode_id) is None
    assert _ep_uuid(t.tenant_id, b.episode_id) is not None

    # Postgres episode A is gone; B remains.
    with tenant_conn(t.tenant_id) as conn:
        eps = {str(r.episode_id) for r in conn.execute(
            sql("SELECT episode_id FROM episodes WHERE tenant_id=:t"),
            {"t": str(t.tenant_id)}).all()}
    assert eps == {b.episode_id}


# --------------------------------------------------------------------------- #
# (2) legal hold blocks eviction (409 through the endpoint)
# --------------------------------------------------------------------------- #
def test_legal_hold_blocks_evict(make_tenant, client, use_scripted):
    t = make_tenant()
    use_scripted(ScriptedLLMClient().add_script(
        "Alice joined Acme", entities=["Alice", "Acme"],
        edges=[{"source": "Alice", "target": "Acme", "relation": "WORKS_AT",
                "fact": "Alice works at Acme"}]))
    a = ingest_text(t.tenant_id, text="Alice joined Acme", occurred_at=JAN,
                    source="note", meta=None)

    r = client.post(f"/v1/admin/episodes/{a.episode_id}/legal_hold",
                    json={"held": True}, headers=h(t))
    assert r.status_code == 200

    r = client.post(f"/v1/admin/episodes/{a.episode_id}/evict", headers=h(t))
    assert r.status_code == 409

    # The episode and its edge are untouched by the blocked eviction.
    assert _ep_uuid(t.tenant_id, a.episode_id) is not None
    assert _edge_count(t.tenant_id) == 1


# --------------------------------------------------------------------------- #
# (3) clear_tenant: dry-run counts, real clear wipes graph, held lineage stays
# --------------------------------------------------------------------------- #
def test_clear_dry_run_then_wipe_keeps_held(
        make_tenant, tenant_conn, use_scripted):
    t = make_tenant()
    use_scripted(ScriptedLLMClient()
                 .add_script("Alice joined Acme", entities=["Alice", "Acme"],
                             edges=[{"source": "Alice", "target": "Acme",
                                     "relation": "WORKS_AT",
                                     "fact": "Alice works at Acme"}])
                 .add_script("Bob joined Beta", entities=["Bob", "Beta"],
                             edges=[{"source": "Bob", "target": "Beta",
                                     "relation": "WORKS_AT",
                                     "fact": "Bob works at Beta"}]))

    a = ingest_text(t.tenant_id, text="Alice joined Acme", occurred_at=JAN,
                    source="note", meta=None)
    b = ingest_text(t.tenant_id, text="Bob joined Beta", occurred_at=FEB,
                    source="note", meta=None)
    assert _edge_count(t.tenant_id) == 2

    with tenant_conn(t.tenant_id) as conn:
        eviction.set_legal_hold(conn, t.tenant_id, b.episode_id, True,
                                actor="test")

    # Dry-run: counts only, no mutation.
    with tenant_conn(t.tenant_id) as conn:
        plan = eviction.clear_tenant(conn, t.tenant_id, actor="test",
                                     dry_run=True)
    assert plan["held_episodes"] == 1
    assert plan["episodes"] == 1          # only the non-held episode A
    assert plan["facts"] == 1             # held evidence is excluded
    assert plan["entities"] == 0           # conservative projection registry
    assert _edge_count(t.tenant_id) == 2  # untouched

    # Real clear.
    with tenant_conn(t.tenant_id) as conn:
        totals = eviction.clear_tenant(conn, t.tenant_id, actor="test",
                                       dry_run=False)
    assert totals["held_episodes"] == 1
    assert totals["facts"] == 1

    with tenant_conn(t.tenant_id) as conn:
        eps = {str(r.episode_id) for r in conn.execute(
            sql("SELECT episode_id FROM episodes WHERE tenant_id=:t"),
            {"t": str(t.tenant_id)}).all()}
        assert eps == {b.episode_id}      # held episode survives
    # The held episode and its graph edge remain queryable.
    assert _ep_uuid(t.tenant_id, b.episode_id) is not None
    assert _ep_uuid(t.tenant_id, a.episode_id) is None
    assert _edge_count(t.tenant_id) == 1
