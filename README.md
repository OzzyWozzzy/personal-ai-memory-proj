# Temporal Memory Graph

A local-first memory engine that turns unstructured episodes into a temporal
knowledge graph, retrieves grounded context with local ML models, and exposes
memory lineage through a visual governance interface.

## Architecture

| Layer | Technical approach |
|---|---|
| Source ledger | Content-addressed Postgres episodes with `occurred_at` and `recorded_at`, protected by forced RLS |
| Temporal graph | FalkorDB entity edges with `valid_at` / `invalid_at`, contradiction closure, deduplication, and episode provenance |
| Local ML | BGE-small embeddings and Answer.AI ColBERT late-interaction reranking through ONNX/FastEmbed |
| Retrieval | Multi-query hybrid search, reciprocal-rank fusion, graph expansion, source hydration, and hard token budgets |
| Governance | Legal holds, dry-run lineage eviction, source retention policies, and append-only audit events |

Postgres preserves raw evidence and governance state. FalkorDB stores entities,
relationships, and temporal validity. Retrieval combines both instead of
forcing every useful detail into a graph edge.

```text
episode -> immutable ledger -> entity/fact extraction -> temporal graph
   |                                               |
   +--------------- source hydration <------------+
                           |
              dense recall + ColBERT rerank
                           |
                   budgeted context
```

## Run locally

Requires Python 3.12. Postgres and FalkorDB run as embedded local services.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e vendor/graphiti -e '.[dev]'
alembic upgrade head
uvicorn app.main:create_app --factory --port 8321
```

Open [http://localhost:8321/ui](http://localhost:8321/ui). Replay mode is
offline and deterministic but only supports committed extraction fixtures. Set
an OpenAI or Anthropic provider in [.env.example](.env.example) to ingest new
text.

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/remember` | Ingest text, structured data, or a conversation |
| `POST` | `/v1/context` | Build grounded context under a token budget |
| `GET` | `/v1/search` | Search entities, facts, and source passages |
| `GET` | `/v1/graph` | Inspect the graph at a point in time |
| `GET` | `/v1/subjects/{id}/story` | Read an entity's temporal history |
| `POST` | `/v1/admin/episodes/{id}/evict` | Preview or apply provenance-aware deletion |

Machine clients authenticate with `X-API-Key`. A zero-dependency local MCP
bridge is included in `mcp_server/`.

## LoCoMo results

Frozen, category-stratified one-third evaluation across the full extraction,
retrieval, source-hydration, answering, and judging path:

| Category | Correct | Score |
|---|---:|---:|
| Multi-hop | 79/94 | 84.04% |
| Temporal | 97/107 | **90.65%** |
| Open-domain | 19/32 | 59.38% |
| Single-hop | 249/280 | 88.93% |
| **Overall** | **444/513** | **86.55%** |

The session-level run was iterated against LoCoMo and is not a sealed benchmark
or state-of-the-art claim. Protocol, confidence intervals, ablations, and
limitations are in [docs/evaluation.md](docs/evaluation.md).

## Verification

```bash
pytest -q
node tests/frontend_graph_layout_probe.cjs
```

Current gate: **180 tests passed**, plus the headless graph-rendering contract.

## License

[MIT](LICENSE)
