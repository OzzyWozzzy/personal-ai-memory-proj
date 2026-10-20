"""Tests for the v4 embedder and cross-encoder.

``import app.engine`` (via app.engine.clients) pins EMBEDDING_DIM=384 before any
graphiti_core import. The fastembed model is cached locally from a prior
experiment; if it cannot load offline in the sandbox, the two embedding tests
skip-with-reason rather than fail, but the class stays correct.
"""
from __future__ import annotations

import asyncio
import os

import numpy as np
import pytest

from app.engine.clients import (
    FastEmbedEmbedder,
    LateInteractionReranker,
    PassthroughCrossEncoder,
    _as_text_list,
)


def _load_or_skip() -> FastEmbedEmbedder:
    embedder = FastEmbedEmbedder()
    try:
        embedder._ensure_model()
    except Exception as exc:  # offline / model not cached
        pytest.skip(f"fastembed model unavailable offline: {exc}")
    return embedder


# --------------------------------------------------------------------------- #
# Input coercion — the prototype's str() bug
# --------------------------------------------------------------------------- #
def test_as_text_list_wraps_bare_string():
    assert _as_text_list("hello") == ["hello"]


def test_as_text_list_passes_through_list():
    # graphiti calls create(input_data=[text]); the element must NOT be
    # stringified into "['text']".
    assert _as_text_list(["hello"]) == ["hello"]
    assert _as_text_list(["a", "b"]) == ["a", "b"]


# --------------------------------------------------------------------------- #
# FastEmbedEmbedder
# --------------------------------------------------------------------------- #
def test_embed_dim_is_384():
    embedder = _load_or_skip()
    vec = asyncio.run(embedder.create(["the quick brown fox"]))
    assert len(vec) == int(os.environ["EMBEDDING_DIM"]) == 384
    assert all(isinstance(x, float) for x in vec)


def test_create_batch_matches_per_item_create():
    embedder = _load_or_skip()
    texts = ["alpha text", "beta text", "gamma text"]
    batch = asyncio.run(embedder.create_batch(texts))
    assert len(batch) == len(texts)
    assert all(len(v) == 384 for v in batch)
    for i, text in enumerate(texts):
        single = asyncio.run(embedder.create([text]))
        assert single == pytest.approx(batch[i], abs=1e-5)


def test_create_input_string_and_singleton_list_agree():
    embedder = _load_or_skip()
    from_str = asyncio.run(embedder.create("same text"))
    from_list = asyncio.run(embedder.create(["same text"]))
    assert from_str == pytest.approx(from_list, abs=1e-6)


def test_dim_mismatch_fails_loudly(monkeypatch):
    embedder = _load_or_skip()
    # Force the check to disagree so the guard's error path is exercised.
    monkeypatch.setenv("EMBEDDING_DIM", "1024")
    with pytest.raises(RuntimeError, match="dim mismatch"):
        asyncio.run(embedder.create(["trigger the guard"]))


def test_embedder_caches_repeated_batch_inputs():
    class FakeModel:
        def __init__(self):
            self.batches = []

        def embed(self, values):
            self.batches.append(list(values))
            for index, _ in enumerate(values, start=1):
                yield np.asarray([float(index), 0.0], dtype=np.float32)

    model = FakeModel()
    embedder = FastEmbedEmbedder()
    embedder._model = model
    # The dimension guard is independently covered with the real 384-d model.
    embedder._dim_verified = True

    first = asyncio.run(embedder.create_batch(["alpha", "beta"]))
    second = asyncio.run(embedder.create_batch(["beta", "alpha"]))

    assert model.batches == [["alpha", "beta"]]
    assert second == [first[1], first[0]]
    assert embedder.cache_stats() == {
        "entries": 2,
        "hits": 2,
        "misses": 2,
        "limit": 50000,
    }


# --------------------------------------------------------------------------- #
# PassthroughCrossEncoder
# --------------------------------------------------------------------------- #
def test_passthrough_scores_descend_in_input_order():
    ce = PassthroughCrossEncoder()
    passages = ["first", "second", "third"]
    ranked = asyncio.run(ce.rank("q", passages))
    assert [p for p, _ in ranked] == passages  # order preserved
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)  # strictly descending


def test_passthrough_handles_empty():
    ce = PassthroughCrossEncoder()
    assert asyncio.run(ce.rank("q", [])) == []


def test_late_interaction_reranks_by_query_relevance():
    reranker = LateInteractionReranker()
    try:
        reranker._ensure_model()
    except Exception as exc:
        pytest.skip(f"late-interaction model unavailable offline: {exc}")
    passages = ["Bob likes chocolate cake", "Ada currently lives in Lisbon"]
    ranked = asyncio.run(reranker.rank("Where does Ada live?", passages))
    assert ranked[0][0] == "Ada currently lives in Lisbon"
    assert ranked[0][1] > ranked[1][1]


def test_late_interaction_caches_passage_token_embeddings():
    class FakeModel:
        def __init__(self):
            self.passage_batches = []

        def query_embed(self, values):
            yield np.asarray([[1.0, 0.0]], dtype=np.float32)

        def embed(self, values):
            self.passage_batches.append(list(values))
            for index, _ in enumerate(values, start=1):
                yield np.asarray([[1.0 / index, 0.0]], dtype=np.float32)

    model = FakeModel()
    reranker = LateInteractionReranker()
    reranker._model = model
    passages = ["one stable passage", "another stable passage"]

    asyncio.run(reranker.rank("first query", passages))
    asyncio.run(reranker.rank("second query", passages))

    assert model.passage_batches == [passages]
    assert reranker.cache_stats() == {
        "entries": 2,
        "hits": 2,
        "misses": 2,
        "limit": 20000,
    }
