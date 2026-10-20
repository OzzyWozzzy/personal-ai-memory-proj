"""End-to-end tests for the single source-ledger -> Graphiti write path."""
from __future__ import annotations

import app.engine  # noqa: F401 - pins Graphiti's embedding dimension
import app.engine.graph as graph

from datetime import datetime, timezone

import pytest
from graphiti_core.nodes import EpisodeType
from sqlalchemy import text

from app.engine.clients import FastEmbedEmbedder, PassthroughCrossEncoder
from app.engine.ingest import _episode_type, ingest_text
from app.engine.rebuild import rebuild_tenant
from tests.engine_fakes import ScriptedLLMClient

UTC = timezone.utc
JAN = datetime(2026, 1, 5, tzinfo=UTC)
FEB = datetime(2026, 2, 10, tzinfo=UTC)


@pytest.fixture(scope="session")
def embedder():
    return FastEmbedEmbedder()


@pytest.fixture
def use_scripted(tmp_path, monkeypatch, embedder):
    monkeypatch.setenv("MEMORY_GRAPH_DIR", str(tmp_path / "graphdb"))

    def configure(client: ScriptedLLMClient):
        graph.configure_clients(
            llm_client=client,
            embedder=embedder,
            cross_encoder=PassthroughCrossEncoder(),
        )
        return client

    yield configure
    graph.close_all()


def headers(tenant):
    return {"X-API-Key": tenant.api_key}


def edge_count(tenant_id):
    engine = graph.get_engine(str(tenant_id))

    async def count():
        rows, _, _ = await engine.driver.execute_query(
            "MATCH ()-[e:RELATES_TO]->() RETURN count(e) AS n"
        )
        return rows[0]["n"]

    return graph._run_coro(count())


def test_episode_type_preserves_actor_turn_semantics():
    assert _episode_type("email") is EpisodeType.text
    assert _episode_type("chat") is EpisodeType.message


def test_ingest_stores_source_extracts_edge_and_projects_entities(
    make_tenant, tenant_conn, use_scripted
):
    tenant = make_tenant()
    use_scripted(
        ScriptedLLMClient().add_script(
            "Alice joined Acme",
            entities=["Alice", "Acme"],
            edges=[{
                "source": "Alice", "target": "Acme", "relation": "WORKS_AT",
                "fact": "Alice works at Acme", "valid_at": JAN.isoformat(),
            }],
        )
    )
    report = ingest_text(
        tenant.tenant_id,
        text="Alice joined Acme",
        occurred_at=JAN,
        source="note",
        meta=None,
    )
    assert report.deduped is False
    assert [edge.fact for edge in report.edges] == ["Alice works at Acme"]
    with tenant_conn(tenant.tenant_id) as conn:
        source = conn.execute(
            text("SELECT content FROM episodes WHERE episode_id=:id"),
            {"id": report.episode_id},
        ).scalar_one()
        names = set(conn.execute(text("SELECT canonical_name FROM entities")).scalars())
    assert source == "Alice joined Acme"
    assert names == {"Alice", "Acme"}
    assert edge_count(tenant.tenant_id) == 1


def test_identical_source_is_idempotent(make_tenant, use_scripted):
    tenant = make_tenant()
    use_scripted(
        ScriptedLLMClient().add_script(
            "Bram bought a canoe",
            entities=["Bram", "canoe"],
            edges=[{
                "source": "Bram", "target": "canoe", "relation": "OWNS",
                "fact": "Bram owns a canoe",
            }],
        )
    )
    first = ingest_text(
        tenant.tenant_id, text="Bram bought a canoe", occurred_at=JAN,
        source="note", meta=None,
    )
    second = ingest_text(
        tenant.tenant_id, text="Bram bought a canoe", occurred_at=FEB,
        source="note", meta=None,
    )
    assert second.deduped is True
    assert second.episode_id == first.episode_id
    assert second.edges == []
    assert edge_count(tenant.tenant_id) == 1


def test_graphiti_native_invalidation_is_audited(
    make_tenant, tenant_conn, use_scripted
):
    tenant = make_tenant()
    scripted = ScriptedLLMClient(
        edge_duplicate={"duplicate_facts": [], "contradicted_facts": [0]}
    )
    scripted.add_script(
        "Kai works at Initech", entities=["Kai", "Initech"],
        edges=[{
            "source": "Kai", "target": "Initech", "relation": "WORKS_AT",
            "fact": "Kai works at Initech", "valid_at": JAN.isoformat(),
        }],
    ).add_script(
        "Kai moved to Initrode", entities=["Kai", "Initrode"],
        edges=[{
            "source": "Kai", "target": "Initrode", "relation": "WORKS_AT",
            "fact": "Kai works at Initrode", "valid_at": FEB.isoformat(),
        }],
    )
    use_scripted(scripted)
    first = ingest_text(
        tenant.tenant_id, text="Kai works at Initech", occurred_at=JAN,
        source="note", meta=None,
    )
    second = ingest_text(
        tenant.tenant_id, text="Kai moved to Initrode", occurred_at=FEB,
        source="note", meta=None,
    )
    assert first.edges[0].uuid in {edge.uuid for edge in second.invalidated_edges}
    with tenant_conn(tenant.tenant_id) as conn:
        detail = conn.execute(
            text("SELECT detail FROM audit_log WHERE action='graph.edge_invalidated'")
        ).scalar_one()
    assert detail["edge_uuid"] == first.edges[0].uuid


def test_remember_retains_no_edge_source_without_parallel_note_store(
    make_tenant, tenant_conn, client, use_scripted
):
    tenant = make_tenant()
    source = "Nordwind cares more about latency than price."
    use_scripted(
        ScriptedLLMClient().add_script(source, entities=["Nordwind"], edges=[])
    )
    response = client.post(
        "/v1/remember", json={"text": source}, headers=headers(tenant)
    )
    assert response.status_code == 200
    assert response.json()["remembered"] == [source]
    with tenant_conn(tenant.tenant_id) as conn:
        assert conn.execute(text("SELECT count(*) FROM episodes")).scalar_one() == 1
        tables = conn.execute(text(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema='public' AND table_name='semantic_memories'"
        )).scalar_one()
    assert tables == 0


def test_remember_handles_conversation_json_and_document(
    make_tenant, client, use_scripted
):
    tenant = make_tenant()
    use_scripted(
        ScriptedLLMClient()
        .add_script("I moved to Porto", entities=["user", "Porto"], edges=[])
        .add_script("Porto has a river", entities=["Porto"], edges=[])
        .add_script("customer: Rae; plan: Pro", entities=["Rae", "Pro"], edges=[])
        .add_script("Rae leads robotics", entities=["Rae", "robotics"], edges=[])
    )
    conversation = client.post(
        "/v1/remember",
        json={"conversation": [
            {"role": "user", "content": "I moved to Porto"},
            {"role": "assistant", "content": "Porto has a river"},
        ]},
        headers=headers(tenant),
    )
    structured = client.post(
        "/v1/remember", json={"json": {"customer": "Rae", "plan": "Pro"}},
        headers=headers(tenant),
    )
    document = client.post(
        "/v1/remember", json={"document": {"content": "Rae leads robotics"}},
        headers=headers(tenant),
    )
    assert conversation.status_code == structured.status_code == document.status_code == 200
    assert len(conversation.json()["details"]["episode_ids"]) == 2


def test_rebuild_replays_immutable_sources(make_tenant, use_scripted):
    tenant = make_tenant()
    client = ScriptedLLMClient().add_script(
        "Ana founded Vertex",
        entities=["Ana", "Vertex"],
        edges=[{
            "source": "Ana", "target": "Vertex", "relation": "FOUNDED",
            "fact": "Ana founded Vertex", "valid_at": JAN.isoformat(),
        }],
    )
    use_scripted(client)
    ingest_text(
        tenant.tenant_id, text="Ana founded Vertex", occurred_at=JAN,
        source="note", meta=None,
    )
    assert edge_count(tenant.tenant_id) == 1
    summary = rebuild_tenant(str(tenant.tenant_id))
    assert summary["episodes"] == 1
    assert edge_count(tenant.tenant_id) == 1
