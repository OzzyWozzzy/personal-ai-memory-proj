"""Graphiti-native memory engine with source-grounded hybrid retrieval.

Module map: bridge (LLM client and fixture record/replay), clients
(embedder/reranker), graph (per-tenant FalkorDB + Graphiti lifecycle),
ingest (the sole graph write path), native (display-only field helpers),
retrieval (/v1/context), and views (graph/story/stats projections).

EMBEDDING_DIM must be set before ANY graphiti_core import: graphiti reads it
as a module-level constant (embedder/client.py) and pads zero-vectors of that
dimension in search — a mismatch with the real embedder (bge-small, 384) is a
silent ranking bug. Import app.engine before graphiti_core everywhere.
"""
from __future__ import annotations

import os

# bge-small-en-v1.5 (app.engine.clients). setdefault so an explicit env wins.
os.environ.setdefault("EMBEDDING_DIM", "384")
