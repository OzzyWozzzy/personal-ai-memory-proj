"""LLM clients for the engine: offline fixtures, a local subagent bridge,
and opt-in direct provider clients for previously unseen input.

Three concrete clients, all subclassing graphiti's ``LLMClient`` so the driver
can use any of them interchangeably:

- ``BridgeLLMClient`` writes each fully-prepared request to a file queue and
  waits for a subagent worker to answer it (generalizes the LongMemEval
  experiment's ``SubagentLLMClient``). This is the only client that ever
  produces *new* model output.
- ``FixtureLLMClient`` replays committed fixtures and raises on a miss — the
  deterministic, offline path used by tests and re-ingest determinism proofs.
- ``RecordingLLMClient`` wraps the bridge: fixture hit → replay; miss →
  delegate to the bridge, validate, persist the fixture, return. This is how a
  live run populates the fixture corpus.

Why we do NOT reuse graphiti's built-in ``LLMCache`` (all clients are built
with ``cache=False``): its key (``_get_cache_key``) hashes only the messages
and ``self.model`` — it omits ``response_model`` and ``model_size``. Two calls
that share a prompt but request different structured outputs (or different
model tiers) would collide. Our fixture key includes both, so record/replay is
sound where graphiti's cache would silently serve the wrong shape.

JSON repair ladder (design §3.1, failure-mode FM-4): in the old system 15
sessions were silently dropped because a single malformed LLM JSON blob threw
away the whole extraction. Here a bridge answer that fails to parse or validate
is (a) deterministically repaired (fences/trailing-commas/junk-wrapping),
then if still bad (b) re-published ONCE with the validation error appended, and
only then (c) raised. graphiti's own tenacity retry does NOT cover pydantic
``ValidationError`` (see LLMClient._generate_response_with_retry), so this
ladder is the last line of defence and lives entirely here.

IMPORTANT: this module is imported as ``app.engine.bridge`` so that
``app/engine/__init__.py`` runs first and pins ``EMBEDDING_DIM=384`` before any
graphiti_core import below.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from graphiti_core.prompts.models import Message

logger = logging.getLogger(__name__)

# app/engine/bridge.py -> repo root is three parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Fixture filenames use only the first 40 hex chars of the sha256 key. 40 hex
# chars = 160 bits, collision-free in practice for any realistic corpus, and
# short enough to keep directory listings readable.
_KEY_FILENAME_LEN = 40


class LLMBridgeError(RuntimeError):
    """A bridge/record/replay request could not be satisfied."""


class FixtureMissError(LLMBridgeError):
    """A replay-only client had no fixture for the requested key."""


# --------------------------------------------------------------------------- #
# Fixture key
# --------------------------------------------------------------------------- #
def fixture_key(
    messages: list[Message],
    response_model: type[BaseModel] | None,
    model_size: ModelSize,
) -> str:
    """Stable content hash identifying one LLM request.

    Includes ``response_model`` name and ``model_size`` precisely because
    graphiti's own cache key omits them — that omission is the collision bug we
    exist to avoid. ``ensure_ascii=False`` keeps non-Latin prompt text byte-for-
    byte stable rather than escaping it into ``\\uXXXX`` (the hash is over utf-8
    bytes either way, but this keeps the pre-hash blob faithful to the input).
    """
    payload = {
        "messages": [m.model_dump() for m in messages],
        "response_model": response_model.__name__ if response_model else None,
        "model_size": model_size.value,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# JSON repair ladder — pure functions, no queue involved
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _try_json_object(text: str) -> dict[str, Any] | None:
    """Parse ``text`` as JSON; return it only if it is an object (dict).

    graphiti's structured outputs are always JSON objects, so a bare array or
    scalar is treated as a non-answer (returns None) — it will fail validation
    and drive the repair ladder rather than being silently accepted.
    """
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _strip_code_fences(text: str) -> str:
    """Remove a single surrounding markdown code fence, if present."""
    match = _FENCE_RE.match(text)
    return match.group(1) if match else text


def _remove_trailing_commas(text: str) -> str:
    """Drop commas that immediately precede a closing brace/bracket."""
    return _TRAILING_COMMA_RE.sub(r"\1", text)


def _extract_outermost_object(text: str) -> str | None:
    """Return the substring from the first ``{`` to the last ``}`` (inclusive).

    Handles chatter-wrapped answers ("Here you go: { ... } hope that helps").
    Returns None when no plausible object span exists.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def deterministic_repair(text: str) -> dict[str, Any] | None:
    """Best-effort recovery of a JSON object from messy model output.

    Tries the raw text, then fence-stripping, trailing-comma removal, and
    outermost-object extraction, plus their combinations. Returns the first
    parse that yields a dict, else None. Pure and unit-testable — no I/O.
    """
    stripped = _strip_code_fences(text)
    variants: list[str] = [text, stripped]
    for base in (text, stripped):
        variants.append(_remove_trailing_commas(base))
        extracted = _extract_outermost_object(base)
        if extracted is not None:
            variants.append(extracted)
            variants.append(_remove_trailing_commas(extracted))
    for candidate in variants:
        obj = _try_json_object(candidate)
        if obj is not None:
            return obj
    return None


def validate_response(
    parsed: dict[str, Any] | None,
    response_model: type[BaseModel] | None,
) -> tuple[bool, str]:
    """Validate a parsed response against ``response_model``.

    Returns (ok, error_text). A None ``response_model`` accepts any dict (some
    graphiti prompts have no schema). The error text is what gets appended to
    the repair retry so the worker sees exactly what was wrong.
    """
    if parsed is None:
        return False, "response was not valid JSON and could not be repaired"
    if response_model is None:
        return True, ""
    try:
        response_model(**parsed)
    except ValidationError as exc:
        # ``str(ValidationError)`` includes input_value, which may be prompt or
        # extracted-memory content. Type + field location is sufficient for a
        # repair attempt and is safe for logs/telemetry.
        safe_errors = [
            {
                "type": str(item.get("type", "validation_error")),
                "loc": [str(part) for part in item.get("loc", ())],
            }
            for item in exc.errors(include_input=False, include_url=False)
        ]
        return False, json.dumps(safe_errors, separators=(",", ":"))
    return True, ""


# Type of the transport-specific attempt coroutine the ladder drives.
# attempt(is_repair, error_text) -> raw response text.
AttemptFn = Callable[[bool, str | None], Awaitable[str]]


async def resolve_with_repair(
    attempt: AttemptFn,
    response_model: type[BaseModel] | None,
    counters: Counter | None = None,
) -> dict[str, Any]:
    """Run the FM-4 repair ladder over an abstract ``attempt`` transport.

    ``attempt`` performs one round-trip (publish+wait for the bridge; a canned
    reply in tests) and returns the worker's raw text. Kept transport-agnostic
    so both ``BridgeLLMClient`` and unit tests can exercise every rung without a
    live queue.
    """

    def _bump(key: str) -> None:
        if counters is not None:
            counters[key] += 1

    # Rung 0/a: first answer, deterministic repair.
    raw = await attempt(False, None)
    clean = _try_json_object(raw) is not None
    parsed = deterministic_repair(raw)
    ok, err = validate_response(parsed, response_model)
    if ok and parsed is not None:
        _bump("clean" if clean else "repaired_deterministic")
        return parsed

    # Rung b: one retry with the validation error fed back to the worker.
    logger.warning("bridge answer invalid, retrying once: %s", err[:200])
    raw = await attempt(True, err)
    parsed = deterministic_repair(raw)
    ok, err = validate_response(parsed, response_model)
    if ok and parsed is not None:
        _bump("repaired_retry")
        return parsed

    # Rung c: give up loudly. Never silently drop — that was FM-4.
    _bump("failed")
    raise LLMBridgeError(f"bridge response failed the JSON repair ladder: {err[:500]}")


# --------------------------------------------------------------------------- #
# Shared config helper
# --------------------------------------------------------------------------- #
def _bridge_config(name: str) -> LLMConfig:
    """Config whose model names document intent; no API key, no base URL."""
    return LLMConfig(model=name, small_model=name)


def _resolve_fixtures_dir(fixtures_dir: str | os.PathLike | None) -> Path:
    """Fixtures live at repo ``fixtures/graphiti`` unless overridden.

    Resolution order: explicit argument > ``MEMORY_FIXTURES_DIR`` env (tests
    point this at a tmp dir) > committed default. Fixtures are COMMITTED, so the
    default is a real repo path, not a scratch dir.
    """
    if fixtures_dir is not None:
        return Path(fixtures_dir)
    env = os.environ.get("MEMORY_FIXTURES_DIR")
    if env:
        return Path(env)
    return REPO_ROOT / "fixtures" / "graphiti"


def _write_json_atomic(path: Path, obj: Any) -> None:
    """Write JSON via tmp + rename so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{uuid.uuid4().hex}_{path.name}")
    _write_private_text(tmp, json.dumps(obj, indent=1, ensure_ascii=False))
    tmp.rename(path)


def _write_private_text(path: Path, value: str) -> None:
    """Create a prompt-bearing file readable only by its process owner."""
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)


# --------------------------------------------------------------------------- #
# BridgeLLMClient
# --------------------------------------------------------------------------- #
class BridgeLLMClient(LLMClient):
    """File-queue bridge: each request becomes ``pending/req_<id>.json``; a
    subagent worker answers it as ``done/<id>.json``.

    The public ``generate_response`` (base class) has already appended the
    response-model JSON schema to the last message before ``_generate_response``
    runs, so the messages we queue are exactly what the worker must answer. We
    also embed ``response_schema`` in the request file for worker self-checking.
    """

    def __init__(
        self,
        bridge_dir: str | os.PathLike | None = None,
        timeout_s: float | None = None,
        poll_interval: float | None = None,
    ):
        super().__init__(_bridge_config("subagent-bridge"), cache=False)
        root = Path(bridge_dir) if bridge_dir is not None else Path(
            os.environ.get("MEMORY_BRIDGE_DIR", str(REPO_ROOT / ".bridge"))
        )
        self.bridge_dir = root
        self.pending_dir = root / "pending"
        self.claimed_dir = root / "claimed"
        self.done_dir = root / "done"
        for d in (self.pending_dir, self.claimed_dir, self.done_dir):
            d.mkdir(parents=True, exist_ok=True, mode=0o700)
            d.chmod(0o700)

        self.timeout_s = (
            timeout_s
            if timeout_s is not None
            else float(os.environ.get("MEMORY_BRIDGE_TIMEOUT_S", "1200"))
        )
        self.poll_interval = (
            poll_interval
            if poll_interval is not None
            else float(os.environ.get("MEMORY_BRIDGE_POLL_S", "0.5"))
        )
        # Belt-and-suspenders for a worker that does not write atomically: after
        # the done file appears, tolerate a brief mid-write window. Three
        # identical observations (the first read plus two quiet intervals) are
        # required before malformed-but-stable text reaches the repair ladder;
        # one interval is too short when a writer pauses between chunks.
        self._mid_write_retries = 40
        self._mid_write_interval = 0.05
        self._mid_write_stable_intervals = 2
        # Repair-ladder rung tallies for observability.
        self.repair_stats: Counter = Counter()

    def _publish(self, request: dict[str, Any]) -> None:
        """Atomically publish a request into ``pending/`` (tmp + rename)."""
        rid = request["id"]
        tmp = self.pending_dir / f".tmp_req_{rid}.json"
        _write_private_text(
            tmp, json.dumps(request, indent=1, ensure_ascii=False))
        tmp.rename(self.pending_dir / f"req_{rid}.json")

    def _discard_request_files(self, rid: str, *, completed: bool) -> None:
        """Minimize full-prompt queue retention after success/cancellation."""
        (self.pending_dir / f"req_{rid}.json").unlink(missing_ok=True)
        if completed:
            (self.claimed_dir / f"req_{rid}.json").unlink(missing_ok=True)
            (self.claimed_dir / f"{rid}.json").unlink(missing_ok=True)
            (self.done_dir / f"{rid}.json").unlink(missing_ok=True)

    async def _await_and_read(self, path: Path) -> str:
        """Wait for ``path`` to appear (within timeout), then read its text.

        Readiness is file existence (workers rename atomically); the extra
        mid-write loop only guards non-atomic writers. We do NOT gate on JSON
        validity here — messy-but-complete output must reach the repair ladder,
        not be mistaken for a still-being-written file.
        """
        deadline = time.monotonic() + self.timeout_s
        while not path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"bridge request {path.stem} unanswered after {self.timeout_s}s"
                )
            await asyncio.sleep(self.poll_interval)

        last: str | None = None
        stable_intervals = 0
        for _ in range(self._mid_write_retries):
            text = path.read_text()
            if text:
                if _try_json_object(text) is not None:
                    return text  # complete and clean — fast path
                if text == last:
                    stable_intervals += 1
                    if stable_intervals >= self._mid_write_stable_intervals:
                        # Stable but messy — hand to the repair ladder. This is
                        # deliberately a quiet-period check, not a JSON gate.
                        return text
                else:
                    stable_intervals = 0
            else:
                stable_intervals = 0
            last = text
            await asyncio.sleep(self._mid_write_interval)
        # Re-read once after the final sleep so a last-moment writer update is
        # not hidden by the cached value from the preceding poll.
        return path.read_text()

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        base_messages = [{"role": m.role, "content": m.content} for m in messages]
        schema = response_model.model_json_schema() if response_model else None
        # Closure-local so concurrent _generate_response calls never share the
        # previous-id used for repair_of (BridgeLLMClient is used concurrently).
        prev_id: str | None = None

        async def attempt(is_repair: bool, error: str | None) -> str:
            nonlocal prev_id
            rid = uuid.uuid4().hex[:16]
            if is_repair:
                repair_note = (
                    "Your previous response was rejected by the schema validator "
                    f"with this error:\n{error}\n\nReturn ONLY a corrected JSON "
                    "object matching the schema — no prose, no code fences."
                )
                msgs = base_messages + [{"role": "user", "content": repair_note}]
            else:
                msgs = base_messages
            request: dict[str, Any] = {
                "id": rid,
                "model_size": model_size.value,
                "response_schema": schema,
                "messages": msgs,
            }
            if is_repair:
                request["repair_of"] = prev_id
            prev_id = rid
            self._publish(request)
            completed = False
            try:
                result = await self._await_and_read(
                    self.done_dir / f"{rid}.json")
                completed = True
                return result
            finally:
                self._discard_request_files(rid, completed=completed)

        return await resolve_with_repair(attempt, response_model, self.repair_stats)


# --------------------------------------------------------------------------- #
# FixtureLLMClient
# --------------------------------------------------------------------------- #
class FixtureLLMClient(LLMClient):
    """Replay-only client: returns committed fixtures, raises on a miss.

    Deterministic and offline — the default engine mode. A miss is a hard error
    (never a silent live call) whose message names only non-content diagnostics.
    """

    def __init__(self, fixtures_dir: str | os.PathLike | None = None):
        super().__init__(_bridge_config("fixture-replay"), cache=False)
        self.fixtures_dir = _resolve_fixtures_dir(fixtures_dir)

    def _fixture_path(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None,
        model_size: ModelSize,
    ) -> tuple[str, Path]:
        key = fixture_key(messages, response_model, model_size)
        return key, self.fixtures_dir / f"{key[:_KEY_FILENAME_LEN]}.json"

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        key, path = self._fixture_path(messages, response_model, model_size)
        if not path.exists():
            raise FixtureMissError(
                f"no fixture at {path} (key {key}, "
                f"response_model={response_model.__name__ if response_model else None}, "
                f"model_size={model_size.value}, message_count={len(messages)})"
            )
        return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# RecordingLLMClient
# --------------------------------------------------------------------------- #
class RecordingLLMClient(LLMClient):
    """Fixture hit → replay; miss → delegate to ``inner``, validate, record.

    Delegates at the ``_generate_response`` level (not the public
    ``generate_response``) so the schema is not appended twice: graphiti's
    public wrapper has already augmented the messages before we are called, and
    the same augmented messages drive both the fixture key here and on replay.
    The inner bridge already runs the repair ladder; we re-validate before
    writing so a bad answer is never persisted as a fixture.
    """

    def __init__(
        self,
        inner: LLMClient,
        fixtures_dir: str | os.PathLike | None = None,
    ):
        super().__init__(_bridge_config("record-over-bridge"), cache=False)
        self.inner = inner
        self.fixtures_dir = _resolve_fixtures_dir(fixtures_dir)

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, Any]:
        key = fixture_key(messages, response_model, model_size)
        path = self.fixtures_dir / f"{key[:_KEY_FILENAME_LEN]}.json"
        if path.exists():
            return json.loads(path.read_text())

        result = await self.inner._generate_response(
            messages, response_model, max_tokens, model_size
        )
        ok, err = validate_response(result, response_model)
        if not ok:
            raise LLMBridgeError(
                f"inner client returned an invalid response; not recording: {err[:300]}"
            )
        _write_json_atomic(path, result)
        return result


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def get_llm_client() -> LLMClient:
    """Select the client from ``MEMORY_GRAPHITI_LLM``.

    - ``replay`` (default): offline fixtures, raise on miss.
    - ``record``: fixture-or-bridge, persisting new answers as fixtures.
    - ``bridge``: bridge only, no recording.
    - ``anthropic``: direct Anthropic structured extraction.
    - ``openai``: direct OpenAI structured extraction.

    Direct modes are necessary when previously unseen notes, documents, or
    email arrive. Replay remains the safe offline default for tests and demos.
    """
    mode = os.environ.get("MEMORY_GRAPHITI_LLM", "replay").strip().lower()
    if mode == "replay":
        return FixtureLLMClient()
    if mode == "record":
        return RecordingLLMClient(BridgeLLMClient())
    if mode == "bridge":
        return BridgeLLMClient()
    if mode == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "MEMORY_GRAPHITI_LLM=anthropic requires ANTHROPIC_API_KEY"
            )
        from graphiti_core.llm_client.anthropic_client import AnthropicClient

        model = (
            os.environ.get("MEMORY_GRAPHITI_MODEL")
            or "claude-haiku-4-5-latest"
        ).strip()
        small_model = (
            os.environ.get("MEMORY_GRAPHITI_SMALL_MODEL") or model
        ).strip()
        return AnthropicClient(
            LLMConfig(api_key=api_key, model=model, small_model=small_model),
            cache=False,
        )
    if mode == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "MEMORY_GRAPHITI_LLM=openai requires OPENAI_API_KEY"
            )
        from graphiti_core.llm_client.openai_client import OpenAIClient

        model = (
            os.environ.get("MEMORY_GRAPHITI_MODEL") or "gpt-4.1-mini"
        ).strip()
        small_model = (
            os.environ.get("MEMORY_GRAPHITI_SMALL_MODEL") or "gpt-4.1-nano"
        ).strip()
        return OpenAIClient(
            LLMConfig(api_key=api_key, model=model, small_model=small_model),
            cache=False,
        )
    raise ValueError(
        "MEMORY_GRAPHITI_LLM must be 'replay', 'record', 'bridge', "
        f"'anthropic', or 'openai', got {mode!r}"
    )
