"""Rebuild a tenant's graph by replaying its Postgres episodes.

``python -m app.engine.rebuild <tenant_id>``

Wipes the tenant's FalkorDB graph, then re-ingests every stored episode in
chronological order (``occurred_at``, ``recorded_at``, ``episode_id``) through the
graph half of the write path. This is BOTH the migration story for existing
workspaces (episodes are the immutable ground truth) AND the determinism proof:
the same episodes replayed through the same recorded LLM fixtures produce the
same graph. Episodes are NOT re-inserted — they already exist; only the graph is
rebuilt.

Run with ``MEMORY_GRAPHITI_LLM=replay`` (the default) so every add_episode is
served from committed fixtures and no bridge/API call is made.
"""
from __future__ import annotations

import sys

from sqlalchemy import text

from app.db import tenant_connection
from app.engine.ingest import flatten_json_text, ingest_episode_into_graph
from app.engine import graph as _graph


def _episode_text(row) -> str:
    """The text handed to graph extraction for one episode row: raw content, or
    a flattened rendering of a structured payload."""
    if row.content is not None:
        return row.content
    if row.content_json is not None:
        return flatten_json_text(row.content_json)
    return ""


def rebuild_tenant(tenant_id: str) -> dict:
    """Wipe and replay one tenant's episodes. Returns a summary dict."""
    tenant_id = str(tenant_id)
    print(f"[rebuild] wiping graph for tenant {tenant_id}")
    from app.db import tenant_graph_write_lock

    with tenant_graph_write_lock(tenant_id):
        _graph.wipe(tenant_id)

    with tenant_connection(tenant_id) as conn:
        rows = conn.execute(
            text(
                "SELECT episode_id, source, occurred_at, content, content_json, meta"
                " FROM episodes WHERE tenant_id = :t"
                " ORDER BY occurred_at, recorded_at, episode_id"
            ),
            {"t": tenant_id},
        ).all()

    total = len(rows)
    print(f"[rebuild] {total} episode(s) to replay")

    edges = invalidated = 0
    for ix, row in enumerate(rows, start=1):
        body = _episode_text(row)
        if not body.strip():
            print(f"[rebuild] {ix}/{total} episode {row.episode_id}: empty, skipped")
            continue
        report = ingest_episode_into_graph(
            tenant_id, str(row.episode_id),
            text=body, occurred_at=row.occurred_at,
            source=row.source, meta=row.meta or None)
        edges += len(report.edges)
        invalidated += len(report.invalidated_edges)
        print(f"[rebuild] {ix}/{total} episode {row.episode_id}: "
              f"+{len(report.edges)} edges, {len(report.invalidated_edges)} invalidated")

    summary = {
        "tenant_id": tenant_id,
        "episodes": total,
        "edges": edges,
        "invalidated_edges": invalidated,
    }
    print(f"[rebuild] done: {summary}")
    return summary


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m app.engine.rebuild <tenant_id>", file=sys.stderr)
        return 2
    rebuild_tenant(argv[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
