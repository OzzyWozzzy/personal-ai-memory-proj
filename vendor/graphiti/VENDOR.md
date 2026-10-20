# Vendored Graphiti

- Upstream: https://github.com/getzep/graphiti
- Pinned: v0.29.2 (ff7e29ccd127d8d9721b5cbb2163a6407ef915fe)
- License: Apache-2.0 (see LICENSE)
- Local patches: marked with `# PATCH(memory-backend):` comments; keep this surface minimal (monthly upstream sync).

Patch inventory (grep for the marker):

- `driver/kuzu_driver.py` — bounded `max_db_size`/`buffer_pool_size`; historical
  (the app engine now runs on FalkorDB; kept to minimize vendor drift).
- `driver/kuzu/operations/entity_edge_ops.py`, `search/search_utils.py` (Kùzu
  edge BFS) — endpoint-neutral incident-edge / both-endpoint BFS; historical,
  Kùzu-only branches.
- `search/search_utils.py::node_distance_reranker` — removed upstream's
  FALKORDB re-zip of records that are already dicts (crashed the
  node-distance reranker with `TypeError` on every FalkorDB call).
- `driver/falkordb_driver.py::build_fulltext_query` — group ids equal to the
  bound graph name are dropped from the RediSearch filter. Hyphenated UUID
  group_ids are tokenized apart by RediSearch TEXT fields, so the escaped
  phrase filter matched nothing and silently returned 0 rows from every BM25
  search; in a graph-per-tenant deployment the filter is tautological anyway.
- `helpers.py`, `graphiti.py`, `driver/driver.py` — removed Graphiti's three
  import-time `load_dotenv()` side effects; Root2 loads `.env.local` and `.env`
  deterministically in `app.environment`.
