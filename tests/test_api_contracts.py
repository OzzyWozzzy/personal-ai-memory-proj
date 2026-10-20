"""Compact contract-smoke for the v4 reimplemented endpoints (plan §5.3).

Not a pipeline test — this asserts the request/response *contracts* the frontend
and MCP clients depend on stay wired after the ledger retirement (Phase 3c):

* ``create_app`` boots and answers ``/healthz``;
* ``POST /v1/context`` clearly retires Allen ``interval_op`` in native mode;
* ``GET /v1/graph`` returns the frontend graph envelope keys;
* ``POST /v1/remember`` 422s on empty text;
* ``GET /v1/stats`` returns the four focused count keys.

Offline + deterministic: the process-wide graph clients are the real
:class:`FastEmbedEmbedder` + a :class:`~tests.engine_fakes.ScriptedLLMClient`, so
endpoints that touch the tenant graph run without any LLM/network. A per-test
``MEMORY_GRAPH_DIR`` sandbox with native teardown keeps graphs isolated.
"""
from __future__ import annotations

# Import-ordering contract: app.engine pins EMBEDDING_DIM=384 before any
# graphiti_core import (see app/engine/__init__.py).
import app.engine  # noqa: F401
import app.engine.graph as graph

import pytest

from tests.engine_fakes import ScriptedLLMClient

from app.engine.clients import FastEmbedEmbedder, PassthroughCrossEncoder
from app.main import create_app

# One real embedder for the module (the ONNX model loads once; it is stateless).
_EMBEDDER = FastEmbedEmbedder()


@pytest.fixture(autouse=True)
def _offline_graph(tmp_path, monkeypatch):
    """Per-test graph root + process-wide offline clients + native teardown —
    what real process startup does, minus the live bridge."""
    monkeypatch.setenv("MEMORY_GRAPH_DIR", str(tmp_path / "graphdb"))
    graph.configure_clients(
        llm_client=ScriptedLLMClient(),
        embedder=_EMBEDDER,
        cross_encoder=PassthroughCrossEncoder(),
    )
    yield
    graph.close_all()


def test_create_app_boots_and_healthz(client):
    assert create_app() is not None            # factory is importable + callable
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_context_retires_interval_op(client, make_tenant):
    t = make_tenant()
    hdr = {"X-API-Key": t.api_key}

    # Any legacy interval request receives a clear 422 migration response.
    bad = client.post("/v1/context", headers=hdr,
                      json={"query": "x", "interval_op": "sometimes",
                            "ref_interval": {"start": "2024-01-01T00:00:00Z"}})
    assert bad.status_code == 422

    missing = client.post("/v1/context", headers=hdr,
                          json={"query": "x", "interval_op": "during"})
    assert missing.status_code == 422

    legacy = client.post("/v1/context", headers=hdr,
                     json={"query": "x", "interval_op": "during",
                           "ref_interval": {"start": "2024-01-01T00:00:00Z",
                                            "end": "2025-01-01T00:00:00Z"}})
    assert legacy.status_code == 422
    assert "Graphiti-native" in legacy.text


def test_graph_envelope_keys(client, make_tenant):
    t = make_tenant()
    r = client.get("/v1/graph", headers={"X-API-Key": t.api_key})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"as_of", "at", "include_historical", "nodes", "edges",
                         "counts"}
    assert set(body["counts"]) == {
        "nodes", "edges", "entities", "episodes", "episodes_total", "sources"
    }


def test_remember_422_on_empty_text(client, make_tenant):
    t = make_tenant()
    hdr = {"X-API-Key": t.api_key}
    assert client.post("/v1/remember", json={}, headers=hdr).status_code == 422
    assert client.post("/v1/remember", json={"text": ""},
                       headers=hdr).status_code == 422
    assert client.post("/v1/remember", json={"text": "   "},
                       headers=hdr).status_code == 422


def test_stats_returns_focused_count_keys(client, make_tenant):
    t = make_tenant()
    r = client.get("/v1/stats", headers={"X-API-Key": t.api_key})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"episodes", "entities", "facts", "facts_current"}
