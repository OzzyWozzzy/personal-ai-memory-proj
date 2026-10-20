"""Per-tenant Graphiti-over-FalkorDB lifecycle.

One FalkorDB *graph* per tenant — the graph name, the Graphiti ``group_id``,
and the tenant UUID are the same string — hosted on a single FalkorDB
deployment. Two connection modes, selected by environment:

* **Server mode**: ``MEMORY_FALKORDB_URL`` set to a
  ``redis://`` / ``rediss://`` / ``unix://`` URL. Many API replicas may share
  one FalkorDB; per-tenant write serialization is provided by the Postgres
  advisory lock in :func:`app.db.tenant_graph_write_lock`, not by this module.
* **Embedded mode** (dev/tests, no Docker): ``MEMORY_FALKORDB_URL`` unset. A
  ``falkordblite`` server (BSD-licensed embedded FalkorDB) is booted under
  ``MEMORY_GRAPH_DIR`` (default ``<repo>/.falkordb``), mirroring the
  ``pgserver`` story for Postgres. Production startup rejects embedded mode
  (:mod:`app.security`).

A future second backend (e.g. Neo4j — ``graphiti_core`` ships ``Neo4jDriver``)
slots in at :func:`_make_client_async` / ``TenantEngine.__init__``; callers
only ever see ``engine.driver`` / ``engine.graphiti`` / ``engine.group_id``.

This module owns three cross-cutting concerns:

1. **One event loop for all graph I/O.** ``redis.asyncio`` connections bind to
   the event loop that created them; driving them from per-call ``asyncio.run``
   loops (the old Kùzu pattern) corrupts the pool. ``_run_coro`` therefore
   submits every coroutine to a single persistent daemon-thread loop, and the
   shared clients are created on that loop. ``app.engine.ingest`` /
   ``app.eviction`` / views / retrieval all funnel through this ``_run_coro``.

2. **Index bootstrap, once per graph.** ``FalkorDriver.build_indices_and_
   constraints`` is real (range + fulltext indices; '`already indexed`' is
   swallowed by the driver), so the Kùzu ``INSTALL FTS``/``_wire_fts`` dance is
   gone. We run it once per (deployment, tenant) per process.

3. **Recipe deep-copy guard.** ``Graphiti.search()`` mutates the shared
   module-level recipe object; ``recipe()`` hands out a deep copy per call so
   concurrent requests never clobber each other's ``limit``.

Clients (LLM / embedder / cross-encoder) are process-wide and tenant-agnostic;
they are injected — this module never imports ``bridge`` or ``clients``. Call
``configure_clients(...)`` once at process start, or pass them to
``get_engine``.
"""
from __future__ import annotations

# Ensure EMBEDDING_DIM is set before graphiti_core is imported below — importing
# this submodule executes app/engine/__init__.py first (Python package import
# order), but we name the dependency explicitly so the ordering is visible and
# cannot be broken by a future re-import.
import app.engine as _engine_pkg  # noqa: F401  (import-ordering contract)

import asyncio
import copy
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

# Graphiti emits its own network telemetry on construction. The engine runs
# offline (fixtures, no API key); disable it before the first Graphiti() so no
# construction ever reaches out. setdefault lets an operator re-enable it.
os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")

from falkordb.asyncio import FalkorDB as _AsyncFalkorDB

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.search import search_config_recipes as _recipes
from graphiti_core.search.search_config import SearchConfig

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_GRAPH_DIR = _REPO_ROOT / ".falkordb"
_EMBEDDED_DB_FILENAME = "graphs.db"


# --- injected process-wide clients -----------------------------------------


@dataclass(frozen=True)
class _Clients:
    llm_client: LLMClient
    embedder: EmbedderClient
    cross_encoder: CrossEncoderClient


_clients: _Clients | None = None
_clients_lock = threading.Lock()


def configure_clients(
    llm_client: LLMClient,
    embedder: EmbedderClient,
    cross_encoder: CrossEncoderClient,
) -> None:
    """Register the process-wide clients ``get_engine`` injects into every new
    ``TenantEngine``. Idempotent-by-overwrite; call once at process start.

    Clients are tenant-agnostic (the bridge LLM client, the shared embedder, and
    the reranker are all per-process), so a single registration serves every
    tenant. Tests inject fakes directly via ``get_engine(..., embedder=...)``.
    """
    global _clients
    with _clients_lock:
        _clients = _Clients(llm_client, embedder, cross_encoder)


def _resolve_clients(
    llm_client: LLMClient | None,
    embedder: EmbedderClient | None,
    cross_encoder: CrossEncoderClient | None,
) -> _Clients:
    """Prefer explicitly-passed clients (tests); otherwise the configured ones."""
    if llm_client is not None and embedder is not None and cross_encoder is not None:
        return _Clients(llm_client, embedder, cross_encoder)
    with _clients_lock:
        configured = _clients
    if configured is None:
        raise RuntimeError(
            "app.engine.graph: no clients configured — call configure_clients() "
            "at process start, or pass llm_client/embedder/cross_encoder to "
            "get_engine()."
        )
    return _Clients(
        llm_client if llm_client is not None else configured.llm_client,
        embedder if embedder is not None else configured.embedder,
        cross_encoder if cross_encoder is not None else configured.cross_encoder,
    )


# --- the one graph event loop ------------------------------------------------

_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_lock = threading.Lock()


def _graph_loop() -> asyncio.AbstractEventLoop:
    """The persistent daemon-thread event loop that owns ALL graph I/O.

    redis.asyncio connections are bound to the loop that created them, so the
    shared FalkorDB clients and every coroutine that uses them must live on one
    loop for the life of the process. The loop (and its thread) intentionally
    survive ``close_all()`` — clients are rebuilt lazily on the same loop.
    """
    global _loop, _loop_thread
    with _loop_lock:
        if _loop is not None and _loop.is_running():
            return _loop
        loop = asyncio.new_event_loop()

        def _spin() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = threading.Thread(target=_spin, name="graph-loop", daemon=True)
        thread.start()
        _loop, _loop_thread = loop, thread
        return loop


def _run_coro(coro):
    """Drive a graph coroutine to completion from synchronous code.

    Submits to the persistent graph loop and blocks for the result, so pooled
    async-redis connections are never shared across event loops. Callers on the
    graph loop's own thread would deadlock — that indicates an async caller
    wrongly dropping into the sync bridge, so it is rejected loudly.
    """
    loop = _graph_loop()
    if threading.current_thread() is _loop_thread:
        raise RuntimeError(
            "graph._run_coro called from the graph loop thread — await the "
            "coroutine directly instead of using the sync bridge"
        )
    return asyncio.run_coroutine_threadsafe(coro, loop).result()


# --- shared FalkorDB clients --------------------------------------------------


def falkordb_url() -> str:
    """External FalkorDB URL; empty string selects embedded mode."""
    return (os.environ.get("MEMORY_FALKORDB_URL") or "").strip()


def _graph_root() -> Path:
    """Embedded-mode storage root (also the engine-cache namespace in tests)."""
    return Path(os.environ.get("MEMORY_GRAPH_DIR") or _DEFAULT_GRAPH_DIR)


def _client_key() -> str:
    url = falkordb_url()
    return url if url else f"embedded::{_graph_root().resolve()}"


_clients_by_key: dict[str, object] = {}
_client_cache_lock = threading.Lock()
# (client_key, tenant_id) pairs whose graph indices were built this process.
_initialized_graphs: set[tuple[str, str]] = set()


async def _make_client_async(key: str):
    """Create the shared async FalkorDB client ON the graph loop.

    Server mode hands the URL to the official client (TLS via ``rediss://``).
    Embedded mode boots a ``falkordblite`` server under the graph root; the
    constructor blocks the loop for the ~1s server start, which is acceptable
    because it happens at most once per (process, root).
    """
    url = falkordb_url()
    if url:
        return _AsyncFalkorDB.from_url(url)
    root = _graph_root().resolve()
    root.mkdir(parents=True, exist_ok=True)
    from redislite import AsyncFalkorDB as _EmbeddedFalkorDB  # dev/test-only dep

    # Background save every 60s when dirty: a hard-killed process (crash,
    # SIGKILL) loses at most a minute of embedded graph writes instead of
    # everything since boot. Clean close still does a final SHUTDOWN SAVE.
    # max_connections: ALL tenants share this one client, so concurrent
    # retrieval fan-out needs more sockets than redis-py's finite default
    # (observed MaxConnectionsError under benchmark concurrency).
    return _EmbeddedFalkorDB(
        str(root / _EMBEDDED_DB_FILENAME),
        serverconfig={"save": "60 1"},
        max_connections=256,
    )


def _get_client():
    """Return the cached shared client for the current connection config."""
    key = _client_key()
    with _client_cache_lock:
        client = _clients_by_key.get(key)
    if client is not None:
        return client
    # Create outside the cache lock (embedded boot takes ~1s), then publish;
    # a racing creator's duplicate embedded handle attaches to the same server
    # (redislite pidfile handshake), so last-write-wins is safe.
    client = _run_coro(_make_client_async(key))
    with _client_cache_lock:
        _clients_by_key.setdefault(key, client)
        return _clients_by_key[key]


# --- one tenant's engine ------------------------------------------------------


class TenantEngine:
    """One tenant's FalkorDriver + Graphiti instance over a shared client.

    The driver binds the tenant's named graph (``database=tenant_id``) on the
    shared FalkorDB connection; ``group_id == tenant_id`` is additionally
    stamped on every node/edge, so ``add_episode(group_id=...)`` never clones.
    ``get_engine``'s cache guarantees one live instance per (config, tenant).
    """

    def __init__(
        self,
        tenant_id: str,
        client,
        *,
        llm_client: LLMClient,
        embedder: EmbedderClient,
        cross_encoder: CrossEncoderClient,
    ):
        self.tenant_id = str(tenant_id)
        self.group_id = str(tenant_id)
        self._closed = False
        # Constructed in a sync context (no running loop on this thread), so
        # FalkorDriver skips its fire-and-forget index task; the explicit,
        # awaited bootstrap below is the ONE place indices are built.
        self._driver: FalkorDriver | None = FalkorDriver(
            falkor_db=client, database=self.group_id
        )
        self._graphiti: Graphiti | None = Graphiti(
            graph_driver=self._driver,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
        )

    @property
    def driver(self) -> FalkorDriver:
        if self._closed or self._driver is None:
            raise RuntimeError(
                f"TenantEngine[{self.tenant_id}] is closed; use get_engine() for a "
                "fresh handle"
            )
        return self._driver

    @property
    def graphiti(self) -> Graphiti:
        if self._closed or self._graphiti is None:
            raise RuntimeError(
                f"TenantEngine[{self.tenant_id}] is closed; use get_engine() for a "
                "fresh handle"
            )
        return self._graphiti

    def close(self) -> None:
        """Detach this engine. The FalkorDB client is shared across tenants and
        deliberately NOT closed here — ``close_all()`` owns client teardown.
        Idempotent; a closed engine raises on use instead of misrouting."""
        self._closed = True
        self._graphiti = None
        self._driver = None


# --- process cache + factory --------------------------------------------------

_engines: dict[tuple[str, str], TenantEngine] = {}
_engines_lock = threading.Lock()


def get_engine(
    tenant_id: str,
    *,
    llm_client: LLMClient | None = None,
    embedder: EmbedderClient | None = None,
    cross_encoder: CrossEncoderClient | None = None,
) -> TenantEngine:
    """Return the cached ``TenantEngine`` for ``tenant_id``, creating it lazily.

    The cache key includes the connection config (URL or embedded root) so a
    changed ``MEMORY_GRAPH_DIR`` (e.g. a test ``tmp_path``) yields a distinct
    engine rather than a stale handle pointing at the old deployment.

    Clients default to the process-wide set registered via ``configure_clients``;
    passing them explicitly (tests) overrides. They are only consulted when an
    engine is *created* — a cache hit returns the existing instance regardless,
    because clients are process-wide constants.
    """
    tenant_id = str(tenant_id)
    key = (_client_key(), tenant_id)
    with _engines_lock:
        engine = _engines.get(key)
        if engine is None:
            clients = _resolve_clients(llm_client, embedder, cross_encoder)
            engine = TenantEngine(
                tenant_id,
                _get_client(),
                llm_client=clients.llm_client,
                embedder=clients.embedder,
                cross_encoder=clients.cross_encoder,
            )
            _engines[key] = engine
        if key not in _initialized_graphs:
            # Range + fulltext indices; the driver swallows 'already indexed',
            # so re-running against a pre-existing graph is a cheap no-op sweep.
            _run_coro(engine.driver.build_indices_and_constraints())
            _initialized_graphs.add(key)
        return engine


def health_check() -> None:
    """Raise if the FalkorDB deployment is unreachable (readiness probe).

    ``GRAPH.LIST`` is a cheap admin command that exercises the real connection
    (and, in embedded mode, boots the server on first call) without touching
    any tenant graph."""
    async def _ping(client) -> None:
        await client.list_graphs()

    _run_coro(_ping(_get_client()))


def graph_exists(tenant_id: str) -> bool:
    """True iff the tenant's named graph exists on the FalkorDB deployment.

    A tenant with no graph writes yet has no graph (indices alone do
    materialize one, so post-``get_engine`` tenants exist until wiped)."""

    async def _exists(client) -> bool:
        return str(tenant_id) in set(await client.list_graphs())

    return bool(_run_coro(_exists(_get_client())))


def close_all() -> None:
    """Close every cached engine and shared client, and clear the caches.

    Graph data stays on the FalkorDB deployment (server data is durable;
    embedded data persists in the graph root) — a subsequent ``get_engine``
    reconnects. The embedded ``falkordblite`` server is shut down so tests and
    process exit never leak a redis-server child."""
    with _engines_lock:
        engines = list(_engines.values())
        _engines.clear()
    for engine in engines:
        engine.close()
    with _client_cache_lock:
        clients = dict(_clients_by_key)
        _clients_by_key.clear()
        _initialized_graphs.clear()

    async def _aclose(key: str, client) -> None:
        if key.startswith("embedded::"):
            # redislite's AsyncFalkorDB.close() marks its sync client
            # _async_managed, which makes the sync _cleanup SKIP both the
            # SHUTDOWN SAVE and the SIGTERM fallback ("let the async wrapper
            # handle shutdown") — but the wrapper never issues either, so a
            # plain close() silently loses all embedded data AND leaks the
            # redis-server child. Shut the server down ourselves through the
            # raw redis connection; redis-py treats the connection dying
            # mid-SHUTDOWN as success.
            raw = getattr(getattr(client, "client", None), "_client", None)
            if raw is not None:
                try:
                    await raw.shutdown(save=True, now=True, force=True)
                except Exception as exc:  # durability failure must be visible
                    print(
                        f"embedded falkordb SHUTDOWN SAVE failed ({exc!r}); "
                        "graph data may not be persisted",
                        file=sys.stderr,
                    )
            # With the server already down, close() just tears down the
            # aioredis connection and the pid/socket files.
            await client.close()
            return
        aclose = getattr(client, "aclose", None)
        if aclose is not None:
            await aclose()

    for key, client in clients.items():
        try:
            _run_coro(_aclose(key, client))
        except Exception:  # pragma: no cover — best-effort teardown
            pass


def wipe(tenant_id: str) -> None:
    """Delete a tenant's graph from the deployment and drop it from the cache.

    Uses FalkorDB's native ``GRAPH.DELETE`` (via ``Graph.delete()``). Idempotent:
    a tenant with no graph is a clean no-op — supports rebuild and clear.
    """
    tenant_id = str(tenant_id)
    key = (_client_key(), tenant_id)
    with _engines_lock:
        engine = _engines.pop(key, None)
    _initialized_graphs.discard(key)
    if engine is not None:
        engine.close()

    async def _delete(client) -> None:
        if str(tenant_id) not in set(await client.list_graphs()):
            return
        await client.select_graph(str(tenant_id)).delete()

    _run_coro(_delete(_get_client()))


# --- search recipes ---------------------------------------------------------

# The named recipes we support. Values are the shared module-level SearchConfig
# objects graphiti ships — NEVER hand these out directly: Graphiti.search()
# mutates the object's `limit` in place, so a shared instance is a cross-request
# data race. `recipe()` returns a deep copy per call.
_RECIPE_TABLE: dict[str, SearchConfig] = {
    "EDGE_HYBRID_SEARCH_RRF": _recipes.EDGE_HYBRID_SEARCH_RRF,
    "NODE_HYBRID_SEARCH_RRF": _recipes.NODE_HYBRID_SEARCH_RRF,
    "EDGE_HYBRID_SEARCH_NODE_DISTANCE": _recipes.EDGE_HYBRID_SEARCH_NODE_DISTANCE,
    "COMBINED_HYBRID_SEARCH_RRF": _recipes.COMBINED_HYBRID_SEARCH_RRF,
    "COMBINED_HYBRID_SEARCH_MMR": _recipes.COMBINED_HYBRID_SEARCH_MMR,
    "COMBINED_HYBRID_SEARCH_CROSS_ENCODER": _recipes.COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
}


def recipe(name: str) -> SearchConfig:
    """Return a fresh deep copy of the named search recipe.

    Nothing in this codebase may pass the shared module-level recipe objects to
    ``Graphiti.search()`` / ``search_()`` — it assigns ``config.limit`` in place,
    so a shared instance leaks one request's limit into another's. Each call here
    yields an independent copy that a caller may mutate freely.
    """
    try:
        base = _RECIPE_TABLE[name]
    except KeyError:
        raise ValueError(
            f"unknown search recipe {name!r}; known: {sorted(_RECIPE_TABLE)}"
        ) from None
    return copy.deepcopy(base)
