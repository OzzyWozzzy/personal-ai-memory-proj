"""Tests for the v4 LLM bridge: fixture key, record/replay, file-queue round
trip, and the FM-4 JSON repair ladder.

No pytest-asyncio in this repo, so async client methods are driven with
``asyncio.run``. ``import app.engine`` MUST precede any graphiti_core import so
EMBEDDING_DIM is pinned first — importing app.engine.bridge satisfies that
(the package __init__ runs before the submodule).
"""
from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import BaseModel

from app.engine import bridge
from app.engine.bridge import (
    BridgeLLMClient,
    FixtureLLMClient,
    FixtureMissError,
    LLMBridgeError,
    RecordingLLMClient,
    deterministic_repair,
    fixture_key,
    resolve_with_repair,
)
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.prompts.models import Message


# --------------------------------------------------------------------------- #
# Response models used to prove the key distinguishes response_model.
# --------------------------------------------------------------------------- #
class ModelA(BaseModel):
    a: int


class ModelB(BaseModel):
    a: int


def _msgs(content: str = "hello") -> list[Message]:
    return [Message(role="system", content="sys"), Message(role="user", content=content)]


# --------------------------------------------------------------------------- #
# Fixture key stability
# --------------------------------------------------------------------------- #
def test_fixture_key_is_stable_for_identical_inputs():
    k1 = fixture_key(_msgs(), ModelA, ModelSize.medium)
    k2 = fixture_key(_msgs(), ModelA, ModelSize.medium)
    assert k1 == k2


def test_fixture_key_differs_by_response_model():
    # Same fields, different class name — graphiti's cache would collide here.
    k_a = fixture_key(_msgs(), ModelA, ModelSize.medium)
    k_b = fixture_key(_msgs(), ModelB, ModelSize.medium)
    assert k_a != k_b


def test_fixture_key_differs_by_model_size():
    k_med = fixture_key(_msgs(), ModelA, ModelSize.medium)
    k_small = fixture_key(_msgs(), ModelA, ModelSize.small)
    assert k_med != k_small


def test_fixture_key_differs_by_messages():
    assert fixture_key(_msgs("x"), ModelA, ModelSize.medium) != fixture_key(
        _msgs("y"), ModelA, ModelSize.medium
    )


# --------------------------------------------------------------------------- #
# Replay hit / miss
# --------------------------------------------------------------------------- #
def test_replay_hit_returns_fixture(tmp_path):
    client = FixtureLLMClient(fixtures_dir=tmp_path)
    key = fixture_key(_msgs(), ModelA, ModelSize.medium)
    (tmp_path / f"{key[:40]}.json").write_text(json.dumps({"a": 7}))
    out = asyncio.run(client._generate_response(_msgs(), ModelA, model_size=ModelSize.medium))
    assert out == {"a": 7}


def test_replay_miss_raises_with_diagnostics(tmp_path):
    client = FixtureLLMClient(fixtures_dir=tmp_path)
    with pytest.raises(FixtureMissError) as exc:
        asyncio.run(client._generate_response(_msgs("needle-content"), ModelA))
    message = str(exc.value)
    assert "needle-content" not in message  # prompt content is never logged
    assert "ModelA" in message  # response_model named


def test_validation_error_does_not_include_model_input():
    ok, detail = bridge.validate_response(
        {"a": "private-memory-value"}, ModelA)
    assert ok is False
    assert "private-memory-value" not in detail
    assert "int_parsing" in detail


# --------------------------------------------------------------------------- #
# Record → replay round trip
# --------------------------------------------------------------------------- #
class _FakeInner:
    """Minimal stand-in for BridgeLLMClient exposing only _generate_response."""

    def __init__(self, result):
        self.result = result
        self.calls = 0

    async def _generate_response(self, messages, response_model=None, max_tokens=0,
                                 model_size=ModelSize.medium):
        self.calls += 1
        return self.result


def test_record_then_replay_round_trip(tmp_path):
    inner = _FakeInner({"a": 5})
    rec = RecordingLLMClient(inner=inner, fixtures_dir=tmp_path)

    # First call: miss -> inner invoked once -> fixture written.
    out1 = asyncio.run(rec._generate_response(_msgs(), ModelA))
    assert out1 == {"a": 5}
    assert inner.calls == 1

    # Second call through the recorder: fixture hit, inner NOT invoked again.
    out2 = asyncio.run(rec._generate_response(_msgs(), ModelA))
    assert out2 == {"a": 5}
    assert inner.calls == 1

    # And a pure replay client reads the very same fixture.
    replay = FixtureLLMClient(fixtures_dir=tmp_path)
    out3 = asyncio.run(replay._generate_response(_msgs(), ModelA))
    assert out3 == {"a": 5}


def test_record_refuses_to_persist_invalid_response(tmp_path):
    inner = _FakeInner({"wrong_field": 1})  # fails ModelA validation
    rec = RecordingLLMClient(inner=inner, fixtures_dir=tmp_path)
    with pytest.raises(LLMBridgeError):
        asyncio.run(rec._generate_response(_msgs(), ModelA))
    assert list(tmp_path.glob("*.json")) == []  # nothing written


# --------------------------------------------------------------------------- #
# File-queue bridge round trip with a fake worker
# --------------------------------------------------------------------------- #
async def _run_with_worker(client: BridgeLLMClient, response_model, worker):
    """Drive one _generate_response concurrently with a fake worker coroutine."""
    gen = client._generate_response(_msgs(), response_model)
    result, _ = await asyncio.gather(gen, worker(client))
    return result


def _claim_next(client: BridgeLLMClient):
    reqs = sorted(client.pending_dir.glob("req_*.json"))
    if not reqs:
        return None
    return json.loads(reqs[0].read_text())


def test_bridge_round_trip(tmp_path):
    client = BridgeLLMClient(bridge_dir=tmp_path, timeout_s=5, poll_interval=0.01)

    async def worker(cl):
        while True:
            req = _claim_next(cl)
            if req is not None:
                (cl.done_dir / f"{req['id']}.json").write_text(json.dumps({"a": 3}))
                return
            await asyncio.sleep(0.01)

    out = asyncio.run(_run_with_worker(client, ModelA, worker))
    assert out == {"a": 3}
    assert not list(client.pending_dir.iterdir())
    assert not list(client.done_dir.iterdir())
    # The published request carried the response schema for worker self-check.
    assert client.repair_stats["clean"] == 1


def test_bridge_timeout_raises(tmp_path):
    client = BridgeLLMClient(bridge_dir=tmp_path, timeout_s=0.2, poll_interval=0.02)
    with pytest.raises(TimeoutError):
        asyncio.run(client._generate_response(_msgs(), ModelA))


def test_bridge_tolerates_mid_write(tmp_path):
    client = BridgeLLMClient(bridge_dir=tmp_path, timeout_s=5, poll_interval=0.01)

    async def worker(cl):
        while True:
            req = _claim_next(cl)
            if req is not None:
                done = cl.done_dir / f"{req['id']}.json"
                done.write_text('{"a":')  # partial — invalid JSON
                await asyncio.sleep(0.06)
                done.write_text(json.dumps({"a": 9}))  # completed
                return
            await asyncio.sleep(0.01)

    out = asyncio.run(_run_with_worker(client, ModelA, worker))
    assert out == {"a": 9}


def test_bridge_forwards_stable_malformed_output_after_bounded_quiet_period(
    tmp_path, monkeypatch
):
    client = BridgeLLMClient(bridge_dir=tmp_path, timeout_s=5, poll_interval=0.01)
    path = client.done_dir / "malformed.json"
    path.write_text("not json, but finished")
    reads = 0
    path_type = type(path)
    original_read_text = path_type.read_text

    def counted_read_text(self, *args, **kwargs):
        nonlocal reads
        reads += 1
        return original_read_text(self, *args, **kwargs)

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(path_type, "read_text", counted_read_text)
    monkeypatch.setattr(bridge.asyncio, "sleep", no_sleep)

    out = asyncio.run(client._await_and_read(path))

    assert out == "not json, but finished"
    # First observation + two quiet intervals, rather than all 40 retries.
    assert reads == 3


# --------------------------------------------------------------------------- #
# JSON repair ladder — pure helpers
# --------------------------------------------------------------------------- #
def test_repair_strips_code_fences():
    assert deterministic_repair('```json\n{"a": 1}\n```') == {"a": 1}


def test_repair_removes_trailing_commas():
    assert deterministic_repair('{"a": 1, "b": [2, 3,],}') == {"a": 1, "b": [2, 3]}


def test_repair_extracts_junk_wrapped_object():
    assert deterministic_repair('Sure! Here it is: {"a": 1} hope that helps') == {"a": 1}


def test_repair_returns_none_on_unrecoverable():
    assert deterministic_repair("not json at all") is None


def test_repair_rejects_non_object_json():
    # A bare array is valid JSON but not a response object.
    assert deterministic_repair("[1, 2, 3]") is None


# --------------------------------------------------------------------------- #
# JSON repair ladder — retry transport (rung b) and double failure (rung c)
# --------------------------------------------------------------------------- #
def test_ladder_retries_on_validation_error_then_succeeds():
    from collections import Counter

    replies = iter(['{"wrong": 1}', '{"a": 42}'])  # bad, then good
    seen_repair = []

    async def attempt(is_repair, error):
        seen_repair.append(is_repair)
        return next(replies)

    counters: Counter = Counter()
    out = asyncio.run(resolve_with_repair(attempt, ModelA, counters))
    assert out == {"a": 42}
    assert seen_repair == [False, True]  # second attempt flagged as repair
    assert counters["repaired_retry"] == 1


def test_ladder_double_failure_raises():
    from collections import Counter

    async def attempt(is_repair, error):
        return '{"still": "wrong"}'  # never satisfies ModelA

    counters: Counter = Counter()
    with pytest.raises(LLMBridgeError):
        asyncio.run(resolve_with_repair(attempt, ModelA, counters))
    assert counters["failed"] == 1


def test_ladder_repairs_fenced_output_without_retry():
    from collections import Counter

    async def attempt(is_repair, error):
        assert not is_repair  # a single deterministic repair must suffice
        return '```json\n{"a": 1}\n```'

    counters: Counter = Counter()
    out = asyncio.run(resolve_with_repair(attempt, ModelA, counters))
    assert out == {"a": 1}
    assert counters["repaired_deterministic"] == 1


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def test_factory_default_is_replay(monkeypatch):
    monkeypatch.delenv("MEMORY_GRAPHITI_LLM", raising=False)
    assert isinstance(bridge.get_llm_client(), FixtureLLMClient)


def test_factory_record_wraps_bridge(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_GRAPHITI_LLM", "record")
    monkeypatch.setenv("MEMORY_BRIDGE_DIR", str(tmp_path))
    client = bridge.get_llm_client()
    assert isinstance(client, RecordingLLMClient)
    assert isinstance(client.inner, BridgeLLMClient)


def test_factory_rejects_unknown_mode(monkeypatch):
    monkeypatch.setenv("MEMORY_GRAPHITI_LLM", "bogus")
    with pytest.raises(ValueError):
        bridge.get_llm_client()


def test_factory_direct_provider_modes_require_keys(monkeypatch):
    monkeypatch.setenv("MEMORY_GRAPHITI_LLM", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        bridge.get_llm_client()

    monkeypatch.setenv("MEMORY_GRAPHITI_LLM", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        bridge.get_llm_client()


def test_factory_builds_direct_provider_clients(monkeypatch):
    from graphiti_core.llm_client.anthropic_client import AnthropicClient
    from graphiti_core.llm_client.openai_client import OpenAIClient

    monkeypatch.setenv("MEMORY_GRAPHITI_LLM", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    monkeypatch.setenv("MEMORY_GRAPHITI_MODEL", "claude-test")
    anthropic = bridge.get_llm_client()
    assert isinstance(anthropic, AnthropicClient)
    assert anthropic.model == "claude-test"

    monkeypatch.setenv("MEMORY_GRAPHITI_LLM", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("MEMORY_GRAPHITI_MODEL", "gpt-test")
    openai = bridge.get_llm_client()
    assert isinstance(openai, OpenAIClient)
    assert openai.model == "gpt-test"


def test_factory_blank_optional_model_values_use_provider_defaults(monkeypatch):
    monkeypatch.setenv("MEMORY_GRAPHITI_MODEL", "")
    monkeypatch.setenv("MEMORY_GRAPHITI_SMALL_MODEL", "")

    monkeypatch.setenv("MEMORY_GRAPHITI_LLM", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    anthropic = bridge.get_llm_client()
    assert anthropic.model == "claude-haiku-4-5-latest"
    assert anthropic.small_model == "claude-haiku-4-5-latest"

    monkeypatch.setenv("MEMORY_GRAPHITI_LLM", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    openai = bridge.get_llm_client()
    assert openai.model == "gpt-4.1-mini"
    assert openai.small_model == "gpt-4.1-nano"
