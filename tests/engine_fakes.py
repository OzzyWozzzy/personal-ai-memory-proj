"""Shared test fakes for the v4 engine (plan §3.4/§6).

The centerpiece is :class:`ScriptedLLMClient`, a deterministic offline LLM stand-in
that other engine test modules reuse. It subclasses graphiti's ``LLMClient`` and
dispatches on ``response_model.__name__``, returning minimal-valid canned
responses whose field shapes match the real pydantic models in
``graphiti_core/prompts/`` exactly.

Entity/edge extraction is CONTENT-ADDRESSED: each script carries a ``match``
substring; when an extraction prompt (which embeds the episode text) contains it,
that script's entities/edges are returned. This makes a given episode extract the
same way regardless of call order — the property the rebuild/determinism proof
needs. Dedupe, timestamp, and summary calls return safe no-op defaults (no
duplicates, no contradictions, no new dates) so a fresh graph never fabricates
merges. An unknown ``response_model`` raises loudly so any new pipeline call
surfaces in tests rather than silently returning junk.

The embedder in engine ingest tests is the REAL ``FastEmbedEmbedder`` (offline
ONNX, deterministic) so hybrid search / node dedup behave as in production.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig


class ScriptedLLMClient(LLMClient):
    """Deterministic, content-addressed LLM fake. See module docstring."""

    def __init__(
        self,
        scripts: list[dict] | None = None,
        *,
        node_resolutions: dict | None = None,
        edge_duplicate: dict | None = None,
    ):
        super().__init__(
            LLMConfig(model="scripted-test", small_model="scripted-test"),
            cache=False,
        )
        # Each script: {"match": str, "entities": [...], "edges": [...]}.
        self.scripts: list[dict] = list(scripts or [])
        # Override the dedupe responses to flag duplicates/contradictions.
        self._node_resolutions = node_resolutions
        self._edge_duplicate = edge_duplicate
        # Per-model call tally for assertions/debugging.
        self.calls: Counter = Counter()

    # -- configuration ------------------------------------------------------ #
    def add_script(self, match: str, *, entities=(), edges=()) -> "ScriptedLLMClient":
        """Register a content-addressed extraction script. ``entities`` are names
        (or ``{"name":..., "entity_type_id":...}`` dicts); ``edges`` are dicts with
        ``source``/``target``/``relation``/``fact`` and optional ISO ``valid_at``/
        ``invalid_at``. Returns self for chaining."""
        self.scripts.append(
            {"match": match, "entities": list(entities), "edges": list(edges)}
        )
        return self

    # -- helpers ------------------------------------------------------------ #
    _CURRENT_RE = re.compile(
        r"<CURRENT[_ ]MESSAGE>(.*?)</CURRENT[_ ]MESSAGE>", re.DOTALL)

    def _find(self, messages) -> dict | None:
        blob = "\n".join(m.content for m in messages)
        # Graphiti extraction prompts embed PREVIOUS episodes as context, so a
        # naive whole-prompt match would hit an earlier episode's script. Match
        # against the CURRENT MESSAGE section when the prompt has one.
        current = self._CURRENT_RE.search(blob)
        haystack = current.group(1) if current else blob
        for script in self.scripts:
            if script["match"] in haystack:
                return script
        return None

    @staticmethod
    def _entity(spec: Any) -> dict:
        if isinstance(spec, dict):
            return {
                "name": spec["name"],
                # Default entity_type_id is deliberately out of range so the node
                # gets the bare 'Entity' label -> projected entity_type 'thing'.
                "entity_type_id": spec.get("entity_type_id", 999),
                "episode_indices": spec.get("episode_indices", [0]),
            }
        return {"name": str(spec), "entity_type_id": 999, "episode_indices": [0]}

    @staticmethod
    def _edge(spec: dict) -> dict:
        return {
            "source_entity_name": spec["source"],
            "target_entity_name": spec["target"],
            "relation_type": spec.get("relation", "RELATES_TO"),
            "fact": spec["fact"],
            "valid_at": spec.get("valid_at"),
            "invalid_at": spec.get("invalid_at"),
            "episode_indices": spec.get("episode_indices", [0]),
        }

    # -- LLMClient contract ------------------------------------------------- #
    async def _generate_response(
        self,
        messages,
        response_model=None,
        max_tokens=None,
        model_size=None,
    ) -> dict:
        name = response_model.__name__ if response_model is not None else None
        self.calls[name] += 1

        if name == "ExtractedEntities":
            script = self._find(messages)
            entities = script["entities"] if script else []
            return {"extracted_entities": [self._entity(e) for e in entities]}

        if name == "ExtractedEdges":
            script = self._find(messages)
            edges = script["edges"] if script else []
            return {"edges": [self._edge(e) for e in edges]}

        if name == "NodeResolutions":
            # Default: no duplicates. Unresolved nodes default to new upstream.
            return self._node_resolutions or {"entity_resolutions": []}

        if name == "EdgeDuplicate":
            return self._edge_duplicate or {
                "duplicate_facts": [],
                "contradicted_facts": [],
            }

        if name == "EdgeTimestamps":
            return {"valid_at": None, "invalid_at": None}

        if name == "BatchEdgeTimestamps":
            return {"timestamps": []}

        if name == "SummarizedEntities":
            return {"summaries": []}

        if name in ("EntitySummary", "Summary"):
            return {"summary": ""}

        if name == "SummaryDescription":
            return {"description": ""}

        raise NotImplementedError(
            f"ScriptedLLMClient has no canned response for response_model={name!r}. "
            "Add a branch (a new pipeline call has appeared)."
        )
