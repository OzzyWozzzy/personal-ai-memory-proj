"""Lineage-aware deletion for source episodes and their graph evidence.

Deleting an episode is a governed cascade that respects legal hold, removes the
episode's Postgres ground truth,
cascades the deletion into the per-tenant graph, and writes one append-only
``eviction.apply`` audit row carrying the counts (never the deleted content
beyond ids).

The graph is authoritative for facts. Evicting episode E:

* deletes the graph ``Episodic`` node named after E's ``episode_id`` (graph
  episodes are named after the Postgres episode id);
* deletes every ``RELATES_TO`` entity edge whose ``episodes`` list is evidenced
  by ONLY that graph episode — an edge also evidenced by another episode
  SURVIVES, with E's graph-episode uuid removed from its ``episodes`` list;
* Postgres episode deletion is authoritative and
  already committed to the transaction, so the graph half runs under the same
  never-fail contract as the write path: a graph failure is audited
  (``graph.dual_write_failed`` + ``graph.rebuild_recommended``) and a rebuild
  recommended, never allowed to sink the Postgres delete.

``clear_tenant`` calls that same path for every non-held episode. Held sources
and their graph evidence remain live; they are never reconstructed after a
destructive whole-graph wipe.

Retention (``tenant_policies.retention`` maps ``source -> max_age_days``):
:func:`retention_sweep` evicts past-age episodes through the same path (held
episodes skipped). Default ``{}`` = keep forever.
"""
from __future__ import annotations

from datetime import timedelta

from sqlalchemy import text

from app.audit import write_audit


class EvictionError(Exception):
    """Base eviction failure (e.g. episode not found)."""


class EvictionBlocked(EvictionError):
    """The episode is under legal hold and cannot be evicted (mapped to 409)."""


# --------------------------------------------------------------------------- #
# sync <-> async bridge — the graph loop owns ALL graph I/O (redis.asyncio
# connections are loop-bound, so per-call asyncio.run loops would corrupt the
# shared client pool).
# --------------------------------------------------------------------------- #
def _run_coro(coro):
    from app.engine import graph as _graph

    return _graph._run_coro(coro)  # noqa: SLF001 — the one sanctioned sync bridge


def _engine(tenant_id: str):
    """The tenant's graph engine, self-configuring the process-wide clients on
    first use (mirrors app.engine.ingest._engine) so eviction works even when it
    is the first graph touch in the process."""
    from app.engine import graph as _graph

    try:
        return _graph.get_engine(str(tenant_id))
    except RuntimeError:
        from app.engine.ingest import _configure_default_clients

        _configure_default_clients()
        return _graph.get_engine(str(tenant_id))


# --------------------------------------------------------------------------- #
# Graph-side provenance deletion.
# --------------------------------------------------------------------------- #
async def _graph_evict_async(engine, episode_name: str, *, dry_run: bool) -> int:
    """Delete E's graph Episodic node and its exclusively-evidenced edges.

    Returns the number of ``RELATES_TO`` relationships evidenced ONLY by this
    episode (which are deleted when ``dry_run`` is False). An edge evidenced by
    another episode survives, with this episode's graph uuid dropped from its
    ``episodes`` list.
    """
    driver = engine.driver
    rows, _, _ = await driver.execute_query(
        "MATCH (ep:Episodic {name: $name}) RETURN ep.uuid AS uuid LIMIT 1",
        name=episode_name)
    if not rows:
        return 0  # never ingested into the graph (deduped, or rebuild pending)
    ep_uuid = rows[0]["uuid"]

    # Edges evidenced ONLY by this episode → the exclusive set (size-1 list).
    cnt, _, _ = await driver.execute_query(
        "MATCH ()-[e:RELATES_TO]->() WHERE $u IN e.episodes "
        "AND size(e.episodes) = 1 RETURN count(e) AS n", u=ep_uuid)
    exclusive = int(cnt[0]["n"]) if cnt else 0
    if dry_run:
        return exclusive

    # (1) Delete the exclusive edges. On FalkorDB an entity edge is a native
    #     relationship, so a plain DELETE removes it.
    await driver.execute_query(
        "MATCH ()-[e:RELATES_TO]->() WHERE $u IN e.episodes "
        "AND size(e.episodes) = 1 DELETE e", u=ep_uuid)
    # (2) Shared edges survive — just drop this episode's uuid from their list.
    shared, _, _ = await driver.execute_query(
        "MATCH ()-[e:RELATES_TO]->() WHERE $u IN e.episodes "
        "AND size(e.episodes) > 1 "
        "RETURN e.uuid AS uuid, e.episodes AS episodes", u=ep_uuid)
    for r in shared:
        remaining = [x for x in r["episodes"] if x != ep_uuid]
        await driver.execute_query(
            "MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() SET e.episodes = $eps",
            uuid=r["uuid"], eps=remaining)
    # (3) The Episodic node itself (+ its MENTIONS / NEXT_EPISODE / HAS_EPISODE
    #     relationships).
    await driver.execute_query(
        "MATCH (ep:Episodic {uuid: $u}) DETACH DELETE ep", u=ep_uuid)
    return exclusive


def _graph_evict(conn, tenant_id, episode_id, actor: str, *, dry_run: bool) -> int:
    """Graph half of an episode eviction. Never-fail: on any failure it audits
    (real runs only) and returns 0 — the Postgres delete is authoritative and
    already applied. Returns the count of graph edges removed (or, for a dry
    run, would be removed)."""
    try:
        engine = _engine(tenant_id)
        if dry_run:  # read-only count; no mutation to serialize
            return _run_coro(_graph_evict_async(engine, str(episode_id),
                                                dry_run=True))
        # Caller holds the tenant's SHARED advisory lock (tenant_connection);
        # graph-write is always acquired second, so lock ordering is uniform.
        from app.db import tenant_graph_write_lock

        with tenant_graph_write_lock(tenant_id):
            return _run_coro(_graph_evict_async(engine, str(episode_id),
                                                dry_run=False))
    except Exception as exc:  # noqa: BLE001 — graph must not sink the ledger delete
        if not dry_run:
            write_audit(conn, tenant_id, actor, "graph.dual_write_failed",
                        {"error": f"{type(exc).__name__}: {exc}",
                         "episode_id": str(episode_id)})
            write_audit(conn, tenant_id, actor, "graph.rebuild_recommended",
                        {"reason": "eviction_graph_delete_failed"})
        return 0


# --------------------------------------------------------------------------- #
# Single-episode eviction
# --------------------------------------------------------------------------- #
def evict_episode(conn, tenant_id, episode_id, *, actor: str,
                  dry_run: bool = False, _graph_delete: bool = True) -> dict:
    """Evict one episode + its graph edges and return affected counts.

    Refuses (``EvictionBlocked``) when the episode is under legal hold. With
    ``dry_run=True`` the identical plan is computed and returned WITHOUT mutating.

    ``_graph_delete=False`` is retained for internal repair tooling.
    """
    ep = conn.execute(
        text("SELECT episode_id, legal_hold FROM episodes"
             " WHERE tenant_id=:t AND episode_id=:e"),
        {"t": str(tenant_id), "e": str(episode_id)}).first()
    if ep is None:
        raise EvictionError(f"episode not found: {episode_id}")
    if ep.legal_hold:
        raise EvictionBlocked(f"episode {episode_id} is under legal hold")

    if dry_run:
        facts = (_graph_evict(conn, tenant_id, episode_id, actor, dry_run=True)
                 if _graph_delete else 0)
        return {"facts": facts, "episodes": 1}

    conn.execute(
        text("DELETE FROM episodes WHERE tenant_id=:t AND episode_id=:e"),
        {"t": str(tenant_id), "e": str(episode_id)})

    facts = (_graph_evict(conn, tenant_id, episode_id, actor, dry_run=False)
             if _graph_delete else 0)

    plan = {"facts": facts, "episodes": 1}
    write_audit(conn, tenant_id, actor, "eviction.apply",
                {"episode_id": str(episode_id), **plan})
    return plan


def clear_tenant(conn, tenant_id, *, actor: str, dry_run: bool = False) -> dict:
    """Evict every non-held episode through the normal lineage-aware path.

    Legal holds are a hard boundary: held source episodes and their graph
    evidence remain live. Entity projection rows are conservative and are not
    removed by a bulk clear because the graph is authoritative for visibility.
    """
    eps = conn.execute(
        text("SELECT episode_id, legal_hold FROM episodes"
             " WHERE tenant_id = :t ORDER BY recorded_at, episode_id"),
        {"t": str(tenant_id)}).all()
    totals = {"facts": 0, "episodes": 0}
    held = 0
    for ep in eps:
        if ep.legal_hold:
            held += 1
            continue
        plan = evict_episode(
            conn, tenant_id, ep.episode_id, actor=actor, dry_run=dry_run
        )
        for k, v in plan.items():
            totals[k] += v
    totals["entities"] = 0
    totals["held_episodes"] = held

    if dry_run:
        return totals

    write_audit(conn, tenant_id, actor, "eviction.clear", dict(totals))
    return totals


def set_legal_hold(conn, tenant_id, episode_id, held: bool, actor: str) -> None:
    """Set/clear the legal-hold flag on one episode. A held episode blocks
    eviction and retention. Raises ``EvictionError`` if the episode is unknown."""
    n = conn.execute(
        text("UPDATE episodes SET legal_hold=:h WHERE tenant_id=:t AND episode_id=:e"),
        {"h": bool(held), "t": str(tenant_id), "e": str(episode_id)}).rowcount
    if n == 0:
        raise EvictionError(f"episode not found: {episode_id}")
    write_audit(conn, tenant_id, actor, "eviction.legal_hold",
                {"episode_id": str(episode_id), "held": bool(held)})


def retention_sweep(conn, tenant_id, *, actor: str, dry_run: bool = True) -> dict:
    """Evict episodes older than their per-source ``max_age_days`` policy.

    ``tenant_policies.retention`` maps ``source -> max_age_days`` (default ``{}``
    = keep forever). Legal-hold episodes are skipped. Returns
    ``{dry_run, policy, swept: {source: {episodes, facts, ...}}}`` — per-source
    aggregate counts. Defaults to ``dry_run=True`` (a planning call).
    """
    row = conn.execute(
        text("SELECT retention FROM tenant_policies WHERE tenant_id=:t"),
        {"t": str(tenant_id)}).first()
    retention = (row.retention if row else None) or {}
    now = conn.execute(text("SELECT now()")).scalar_one()

    swept: dict = {}
    for source, max_age_days in retention.items():
        try:
            days = int(max_age_days)
        except (TypeError, ValueError):
            continue  # malformed policy value — skip this source
        cutoff = now - timedelta(days=days)
        ep_ids = conn.execute(
            text("SELECT episode_id FROM episodes WHERE tenant_id=:t"
                 "   AND source=:s AND occurred_at < :cutoff AND NOT legal_hold"
                 " ORDER BY occurred_at"),
            {"t": str(tenant_id), "s": source, "cutoff": cutoff}).scalars().all()
        agg = {"episodes": 0, "facts": 0}
        for eid in ep_ids:
            p = evict_episode(conn, tenant_id, eid, actor=actor, dry_run=dry_run)
            for k in agg:
                agg[k] += p[k]
        swept[source] = agg

    write_audit(conn, tenant_id, actor, "retention.sweep",
                {"dry_run": dry_run, "swept": swept})
    return {"dry_run": dry_run, "policy": retention, "swept": swept}
