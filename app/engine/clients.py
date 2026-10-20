"""Local embedding and reranking clients for the Graphiti engine.

Embeddings cannot come from the LLM bridge — they must be real vectors — so they
run locally via fastembed (BAAI/bge-small-en-v1.5, 384-dim, ONNX, no API key).
Retrieval uses a compact ColBERT late-interaction model as Graphiti's reranker.
It provides query-aware token-level scoring without PyTorch or a hosted API.

Imported as ``app.engine.clients`` so ``app/engine/__init__.py`` pins
``EMBEDDING_DIM=384`` before graphiti_core is imported below.
"""
from __future__ import annotations

import os
import asyncio
import hashlib
import threading
from collections import OrderedDict
from collections.abc import Iterable

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder.client import EmbedderClient

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
_DEFAULT_RERANK_MODEL = "answerdotai/answerai-colbert-small-v1"


def _as_text_list(input_data: object) -> list[str]:
    """Coerce graphiti's polymorphic embedder input into a list of strings.

    graphiti calls ``create(input_data=[text])`` — a single-element LIST, not a
    bare string (see nodes.py / edges.py / search.py). The LongMemEval prototype
    coerced with ``str(input_data)``, which turns ``['hi']`` into the literal
    ``"['hi']"`` and embeds the wrong text — a silent quality bug. Here a list is
    passed through element-wise; a bare string is wrapped.
    """
    if isinstance(input_data, str):
        return [input_data]
    if isinstance(input_data, (list, tuple)):
        return [item if isinstance(item, str) else str(item) for item in input_data]
    # Any other iterable (defensive; graphiti does not exercise this path).
    if isinstance(input_data, Iterable):
        return [str(item) for item in input_data]
    return [str(input_data)]


class FastEmbedEmbedder(EmbedderClient):
    """Local ONNX embeddings via fastembed.

    The model loads lazily so importing this module stays cheap (fastembed pulls
    in onnxruntime). Both ``create`` and ``create_batch`` are implemented:
    graphiti's dedupe path calls ``create_batch`` and only falls back to
    per-item ``create`` on ``NotImplementedError``, so a proper batch
    implementation is a real throughput win, not optional.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self._model_name = model_name
        self._model = None  # lazy
        self._dim_verified = False
        self._lock = threading.RLock()
        self._cache_limit = int(os.environ.get("MEMORY_EMBED_CACHE", "50000"))
        self._cache: OrderedDict[bytes, list[float]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

    def _ensure_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed import TextEmbedding

                    self._model = TextEmbedding(self._model_name)
        return self._model

    def _verify_dim(self, vector: list[float]) -> None:
        """Fail loudly if the model's dimension disagrees with EMBEDDING_DIM.

        graphiti pads zero-vectors of ``EMBEDDING_DIM`` during search; a mismatch
        with the real embedder is a silent ranking corruption (it bit the prior
        experiment). Checked once, on the first embedding produced.
        """
        if self._dim_verified:
            return
        expected = int(os.environ["EMBEDDING_DIM"])
        if len(vector) != expected:
            raise RuntimeError(
                f"embedding dim mismatch: model {self._model_name!r} produced "
                f"{len(vector)} dims but EMBEDDING_DIM={expected}. Fix the env or "
                f"the model choice — a mismatch silently corrupts hybrid search."
            )
        self._dim_verified = True

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        """Embed a single input, returning one vector.

        graphiti passes ``[text]``; we embed it and return the first (and only)
        vector, matching the OpenAIEmbedder contract (``result.data[0]``).
        """
        texts = _as_text_list(input_data)
        vectors = await self.create_batch(texts)
        return vectors[0]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """Embed many inputs in one model call — the dedupe hot path."""
        if not input_data_list:
            return []
        model = self._ensure_model()
        if self._cache_limit <= 0:
            with self._lock:
                vectors = [[float(x) for x in v] for v in model.embed(input_data_list)]
                self._cache_misses += len(vectors)
        else:
            keys = [
                hashlib.sha256(text.encode("utf-8")).digest()
                for text in input_data_list
            ]
            by_key: dict[bytes, list[float]] = {}
            missing_keys: list[bytes] = []
            missing_texts: list[str] = []
            seen_missing: set[bytes] = set()
            with self._lock:
                for key, text in zip(keys, input_data_list, strict=True):
                    if key in self._cache:
                        self._cache_hits += 1
                        self._cache.move_to_end(key)
                        by_key[key] = self._cache[key]
                    elif key not in seen_missing:
                        seen_missing.add(key)
                        missing_keys.append(key)
                        missing_texts.append(text)
                if missing_texts:
                    missing_vectors = [
                        [float(x) for x in vector]
                        for vector in model.embed(missing_texts)
                    ]
                    self._cache_misses += len(missing_vectors)
                    for key, vector in zip(
                        missing_keys, missing_vectors, strict=True
                    ):
                        by_key[key] = vector
                        self._cache[key] = vector
                        self._cache.move_to_end(key)
                        while len(self._cache) > self._cache_limit:
                            self._cache.popitem(last=False)
                vectors = [by_key[key] for key in keys]
        if vectors:
            self._verify_dim(vectors[0])
        return vectors

    def cache_stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._cache),
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "limit": self._cache_limit,
            }


class PassthroughCrossEncoder(CrossEncoderClient):
    """No-op reranker: returns passages in input order with descending scores.

    The engine's RRF recipes never call ``rank``; this exists so instantiating
    the driver does not reach for an OpenAI cross-encoder. Scores strictly
    descend so any caller that does sort by score preserves input order.
    """

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        n = max(len(passages), 1)
        return [(passage, 1.0 - i / n) for i, passage in enumerate(passages)]


class LateInteractionReranker(CrossEncoderClient):
    """Query-aware local reranking through a compact ColBERT ONNX model.

    Graphiti calls this interface a cross encoder.  ColBERT is technically a
    late-interaction reranker: it embeds query and passage tokens separately and
    applies MaxSim.  It has the same operational role—reranking a broad hybrid
    candidate pool—but is much lighter than the vendored PyTorch BGE client.

    The model is loaded lazily.  fastembed downloads it on first production use
    and then reuses its normal model cache; unit tests inject the passthrough
    client and never require the model.
    """

    def __init__(self, model_name: str = _DEFAULT_RERANK_MODEL):
        self._model_name = model_name
        self._model = None
        self._lock = threading.Lock()
        self._passage_cache_limit = int(
            os.environ.get("MEMORY_RERANK_PASSAGE_CACHE", "20000")
        )
        # Key by digest so the cross-tenant process cache does not retain raw
        # passage text. Values are immutable token-embedding arrays.
        self._passage_cache: OrderedDict[bytes, object] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

    def _ensure_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed import LateInteractionTextEmbedding

                    self._model = LateInteractionTextEmbedding(self._model_name)
        return self._model

    def _rank_sync(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        import numpy as np

        model = self._ensure_model()
        with self._lock:
            query_embedding = next(iter(model.query_embed([query])))
            if self._passage_cache_limit <= 0:
                passage_embeddings = list(model.embed(passages))
                self._cache_misses += len(passage_embeddings)
            else:
                passage_embeddings_by_key: dict[bytes, object] = {}
            keys = [hashlib.sha256(passage.encode("utf-8")).digest() for passage in passages]
            missing_texts: list[str] = []
            missing_keys: list[bytes] = []
            seen_missing: set[bytes] = set()
            if self._passage_cache_limit > 0:
                for key, passage in zip(keys, passages, strict=True):
                    if key in self._passage_cache:
                        self._cache_hits += 1
                        self._passage_cache.move_to_end(key)
                        passage_embeddings_by_key[key] = self._passage_cache[key]
                    elif key not in seen_missing:
                        seen_missing.add(key)
                        missing_keys.append(key)
                        missing_texts.append(passage)

                if missing_texts:
                    missing_embeddings = list(model.embed(missing_texts))
                    self._cache_misses += len(missing_embeddings)
                    for key, embedding in zip(
                        missing_keys, missing_embeddings, strict=True
                    ):
                        passage_embeddings_by_key[key] = embedding
                        self._passage_cache[key] = embedding
                        self._passage_cache.move_to_end(key)
                        while len(self._passage_cache) > self._passage_cache_limit:
                            self._passage_cache.popitem(last=False)
                passage_embeddings = [passage_embeddings_by_key[key] for key in keys]
        scored = []
        for passage, passage_embedding in zip(passages, passage_embeddings, strict=False):
            # ColBERT MaxSim: for every query token, keep the strongest passage
            # token match and sum those matches into one relevance score.
            score = float(np.max(query_embedding @ passage_embedding.T, axis=1).sum())
            scored.append((passage, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def cache_stats(self) -> dict[str, int]:
        """Operational visibility for benchmarks and production diagnostics."""
        with self._lock:
            return {
                "entries": len(self._passage_cache),
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "limit": self._passage_cache_limit,
            }

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        if not passages:
            return []
        return await asyncio.to_thread(self._rank_sync, query, passages)
