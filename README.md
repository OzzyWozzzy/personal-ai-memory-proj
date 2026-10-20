# Temporal Memory Graph

A local-first reference implementation for source-grounded temporal memory.
It turns episodes into a Graphiti/FalkorDB knowledge graph, preserves native
validity windows and invalidations, retrieves with local embedding/reranking
models, and exposes the graph through a small governance-focused inspector.

This repository is deliberately not a SaaS starter. There are no user accounts,
cloud connectors, webhooks, billing paths, or hosted deployment manifests.

## What is technically interesting

| Layer | Implementation |
|---|---|
| Source truth | Immutable, content-addressed Postgres episodes; replayable and protected by forced RLS |
| Temporal graph | Graphiti over FalkorDB; entity deduplication, `valid_at` / `invalid_at`, contradiction handling, and provenance |
| Local ML | BGE-small embeddings plus Answer.AI ColBERT late-interaction reranking through ONNX/FastEmbed |
| Retrieval | Multi-query hybrid search, reciprocal-rank fusion, bidirectional graph expansion, local reranking, exact source hydration, and hard token budgets |
| Governance | Legal holds, dry-run lineage eviction, per-source retention, append-only audit events, and retrieval-token metrics |
| Determinism | Content-addressed LLM fixture record/replay plus graph rebuild from immutable episodes |

The modified approach is intentionally hybrid. The graph is authoritative for
entities, relationships, and temporal state; Postgres is authoritative for raw
source text and governance. Retrieval combines both channels instead of forcing
every useful memory into an edge.

```text
episode -> Postgres ledger -> Graphiti extraction -> FalkorDB
   |                                      |
   +------ exact source hydration <-------+
                     |
          BGE recall + ColBERT rerank
                     |
              budgeted context
```

## Local setup

Requires Python 3.12. Postgres and FalkorDB are embedded for local development;
no Docker or external database service is required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e vendor/graphiti -e '.[dev]'
alembic upgrade head
uvicorn app.main:create_app --factory --port 8321
```

Open [http://localhost:8321/ui](http://localhost:8321/ui). The UI creates one
persistent local workspace using a hash-only credential in an HttpOnly cookie.

`MEMORY_GRAPHITI_LLM=replay` is deterministic and offline, but it can only
ingest text with a recorded fixture. For new text, set `MEMORY_GRAPHITI_LLM` to
`openai` or `anthropic` and provide the corresponding API key. See
[.env.example](.env.example) for the focused configuration surface.

## Core API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/remember` | Ingest one text, JSON object, document, or conversation |
| `POST` | `/v1/context` | Assemble source-grounded context under a token budget |
| `GET` | `/v1/search` | Search entities, edges, and source passages |
| `GET` | `/v1/graph` | Time-sliced graph with optional source-provenance nodes |
| `GET` | `/v1/subjects/{id}/story` | Human-readable temporal history |
| `GET` | `/v1/subjects/{id}/history` | Raw edge history |
| `POST` | `/v1/admin/episodes/{id}/evict` | Preview or apply lineage-aware eviction |
| `POST` | `/v1/admin/episodes/{id}/legal_hold` | Protect or release a source episode |
| `GET/PUT` | `/v1/admin/retention` | Inspect or replace source retention policy |

Machine clients use `X-API-Key`. The local UI can mint a narrow, expiring key
for the included MCP server without exposing its browser credential.

## Evaluation results

The frozen LoCoMo graph-memory evaluation covers the complete path: Graphiti
extraction, temporal graph retrieval, source hydration, answer generation, and
binary J-score judging.

| Category | Correct | Score | Wilson 95% CI |
|---|---:|---:|---:|
| Multi-hop | 79/94 | 84.04% | 75.33–90.08 |
| Temporal | 97/107 | **90.65%** | 83.65–94.84 |
| Open-domain | 19/32 | 59.38% | 42.26–74.48 |
| Single-hop | 249/280 | 88.93% | 84.71–92.09 |
| **Overall** | **444/513** | **86.55%** | **83.32–89.23** |

This is a deterministic, category-stratified one-third sample of LoCoMo
categories 1–4. A direct-source retrieval ablation scored 90.25%, but used a
different answerer model and is not a controlled graph-versus-source
comparison. The graph run uses session-level ingestion and was iterated against
LoCoMo; no state-of-the-art, latency, or ingest-cost claim is made. See
[docs/evaluation.md](docs/evaluation.md) for the protocol, ablations,
reproducibility commands, and limitations.

## Verification

```bash
pytest -q
```

The suite exercises the real embedded Postgres and FalkorDB stores, real local
embedding model, temporal invalidation, RLS isolation, source-grounded
retrieval, reranking, graph rebuild, governance, API contracts, and MCP bridge.
The current clean-tree gate is **180 passed**.

## Repository map

```text
app/engine/       Graphiti clients, ingestion, retrieval, rebuild, views
app/api/          focused HTTP surface and governance
frontend/         dependency-free graph/governance inspector
mcp_server/       thin local MCP-to-HTTP bridge
tests/            unit and embedded integration tests
testdata/locomo/  reproducible evaluation harness
vendor/graphiti/  reviewed Graphiti patch set and provenance
```

Graphiti modifications are recorded in
[vendor/graphiti/VENDOR.md](vendor/graphiti/VENDOR.md). Generated benchmark
runs, local databases, model caches, and credentials are intentionally ignored.

## License

Released under the [MIT License](LICENSE).
