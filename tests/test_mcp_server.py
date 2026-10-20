"""MCP server integration: spawn ``python -m mcp_server`` as a real
subprocess against a live uvicorn instance and drive the JSON-RPC 2.0
exchange over stdin/stdout (initialize -> tools/list -> tools/call).

The uvicorn subprocess reuses the pytest session's already-booted embedded
devdb via POSTGRES_ADMIN_URL / POSTGRES_APP_URL, so it never touches pgserver
itself. ANTHROPIC_API_KEY is stripped so nothing calls a live model.
"""
from __future__ import annotations

import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from app import db
from mcp_server import (
    BackendError,
    MemoryClient,
    _naturalize_block,
    tool_remember,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server(_database):
    port = _free_port()
    env = {**os.environ,
           "POSTGRES_ADMIN_URL": db.get_admin_url(),
           "POSTGRES_APP_URL": db.get_app_url()}
    env.pop("ANTHROPIC_API_KEY", None)  # deterministic offline extraction
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:create_app", "--factory",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=REPO_ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base + "/healthz", timeout=1):
                break
        except OSError:
            if proc.poll() is not None:
                raise RuntimeError(
                    "uvicorn died: " + proc.stderr.read().decode()[-2000:])
            time.sleep(0.2)
    else:
        proc.terminate()
        raise RuntimeError("uvicorn did not come up in 30s")
    yield base
    proc.terminate()
    proc.wait(timeout=10)


class Mcp:
    """Line-oriented JSON-RPC driver for the MCP subprocess."""

    def __init__(self, base_url: str, api_key: str):
        env = {**os.environ, "MEMORY_API_URL": base_url,
               "MEMORY_API_KEY": api_key}
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "mcp_server"], cwd=REPO_ROOT, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1)
        self._lines: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.proc.stdout:
            self._lines.put(line)

    def send_raw(self, raw: str) -> None:
        self.proc.stdin.write(raw + "\n")
        self.proc.stdin.flush()

    def send(self, obj: dict) -> None:
        self.send_raw(json.dumps(obj))

    def recv(self, timeout: float = 30.0) -> dict:
        line = self._lines.get(timeout=timeout)
        msg = json.loads(line)
        assert msg["jsonrpc"] == "2.0"  # valid JSON-RPC framing on every reply
        return msg

    def rpc(self, msg_id, method: str, params: dict | None = None,
            timeout: float = 30.0) -> dict:
        self.send({"jsonrpc": "2.0", "id": msg_id, "method": method,
                   "params": params or {}})
        resp = self.recv(timeout=timeout)
        assert resp["id"] == msg_id
        return resp

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


@pytest.fixture
def mcp(live_server, make_tenant):
    t = make_tenant()
    server = Mcp(live_server, t.api_key)
    yield server
    server.close()


def _tool_text(resp: dict) -> str:
    content = resp["result"]["content"]
    assert content[0]["type"] == "text"
    return content[0]["text"]


@pytest.mark.parametrize("url", [
    "file:///tmp/memory.sock",
    "http://memory.example.com",
    "https://user:password@memory.example.com",
    "https://memory.example.com?key=leak",
])
def test_mcp_rejects_unsafe_backend_urls(url):
    with pytest.raises(BackendError):
        MemoryClient(url, "test-key")


def test_mcp_allows_loopback_http_and_remote_https():
    assert MemoryClient("http://127.0.0.1:8321", "key").base.startswith("http://")
    assert MemoryClient("https://memory.example.com", "key").base.startswith("https://")


def test_remember_maps_when_to_backend_occurred_at():
    calls = []

    class StubClient:
        def request(self, method, path, *, body=None, params=None):
            calls.append((method, path, body, params))
            return 201, {"stored": True}

    result = tool_remember(StubClient(), {
        "text": "The launch is on 2026-09-15.",
        "when": "2026-09-15T00:00:00Z",
    })

    assert result == "Saved."
    assert calls == [(
        "POST",
        "/v1/remember",
        {
            "text": "The launch is on 2026-09-15.",
            "occurred_at": "2026-09-15T00:00:00Z",
        },
        None,
    )]


def test_naturalize_block_strips_scaffolding():
    """The context block must read as plain sentences, not a record dump:
    ids, fact:/source:/entity: prefixes, pipes, and 'sources:N' are removed
    while the statements, validity windows, and summaries are kept."""
    block = "\n".join([
        "fact:87bd0eaf | Jane Okafor lives in Kaohsiung | since 2026-05-02 | sources:1",
        "fact:df3c24af | Churn means a customer canceling their subscription | timeless | sources:1",
        "source:1fee1bdf | email | 2026-05-02 | Heads up: Jane loves the new office.",
        "entity:Jane Okafor | SRE lead who relocated from Taipei to Kaohsiung.",
    ])
    out = _naturalize_block(block)

    for scaffold in ("fact:", "source:", "entity:", "sources:", "87bd0eaf",
                     "1fee1bdf", " | "):
        assert scaffold not in out, f"scaffolding leaked: {scaffold!r}"

    # Content is preserved and made readable.
    assert "Jane Okafor lives in Kaohsiung (since 2026-05-02)." in out
    # A timeless window is dropped, not rendered as "(timeless)".
    assert "Churn means a customer canceling their subscription." in out
    assert "(timeless)" not in out
    # Source lines keep only their text (role/date/id dropped); a colon inside
    # the text survives.
    assert "Heads up: Jane loves the new office." in out
    assert "Jane Okafor — SRE lead who relocated from Taipei to Kaohsiung." in out


# ---------------------------------------------------------------------------
def test_full_exchange(mcp):
    # initialize
    resp = mcp.rpc(1, "initialize", {
        "protocolVersion": "2025-06-18",
        "capabilities": {}, "clientInfo": {"name": "pytest", "version": "0"}})
    assert resp["result"]["protocolVersion"] == "2025-06-18"
    assert resp["result"]["serverInfo"]["name"] == "temporal-memory"
    assert resp["result"]["serverInfo"]["title"] == "Temporal Memory Graph"
    assert "tools" in resp["result"]["capabilities"]

    # notifications/initialized: a notification — no response may be sent.
    mcp.send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    # tools/list: 4 tools with plain descriptions.
    resp = mcp.rpc(2, "tools/list")
    tools = resp["result"]["tools"]
    assert {t["name"] for t in tools} == \
        {"remember", "search_memory", "get_context", "get_story"}
    for tool in tools:
        assert "temporal" in tool["title"].lower()
        assert len(tool["description"]) > 20
        assert tool["inputSchema"]["type"] == "object"
        low = tool["description"].lower()
        assert not any(w in low for w in ("tenant", "uuid", "rrf", "jsonb"))
    ctx_tool = next(t for t in tools if t["name"] == "get_context")
    assert "before answering" in ctx_tool["description"]
    remember_tool = next(t for t in tools if t["name"] == "remember")
    assert remember_tool["annotations"]["readOnlyHint"] is False
    assert remember_tool["annotations"]["destructiveHint"] is False
    for tool in tools:
        if tool["name"] != "remember":
            assert tool["annotations"]["readOnlyHint"] is True
            assert tool["annotations"]["destructiveHint"] is False

    # tools/call remember -> readable confirmation text.
    resp = mcp.rpc(3, "tools/call", {
        "name": "remember",
        "arguments": {"text": "Priya Sharma moved to the Platform team on "
                              "2026-03-01."}})
    text = _tool_text(resp)
    assert text.strip()
    assert not resp["result"].get("isError")
    assert "Priya Sharma" in text or "Saved" in text or "Couldn't save" in text

    # tools/call get_context -> non-empty plain text block.
    resp = mcp.rpc(4, "tools/call", {
        "name": "get_context",
        "arguments": {"query": "What do we know about Priya Sharma?",
                      "max_words": 400}})
    text = _tool_text(resp)
    assert text.strip()
    assert not resp["result"].get("isError")


def test_parse_error_keeps_loop_alive(mcp):
    mcp.send_raw("this is not json{")
    resp = mcp.recv()
    assert resp["error"]["code"] == -32700
    assert resp["id"] is None
    # The loop survives: a normal request still answers.
    resp = mcp.rpc(10, "ping")
    assert resp["result"] == {}


def test_unknown_method_and_unknown_tool(mcp):
    resp = mcp.rpc(20, "resources/list")
    assert resp["error"]["code"] == -32601

    resp = mcp.rpc(21, "tools/call", {"name": "frobnicate", "arguments": {}})
    assert resp["result"]["isError"] is True
    assert "Unknown tool" in _tool_text(resp)


def test_missing_api_key_is_tool_error_not_crash(live_server):
    server = Mcp(live_server, "")
    try:
        resp = server.rpc(1, "initialize", {"protocolVersion": "2025-06-18"})
        assert resp["result"]["serverInfo"]["name"] == "temporal-memory"
        resp = server.rpc(2, "tools/call", {
            "name": "get_context", "arguments": {"query": "anything"}})
        assert resp["result"]["isError"] is True
        assert "MEMORY_API_KEY" in _tool_text(resp)
        resp = server.rpc(3, "ping")  # still alive
        assert resp["result"] == {}
    finally:
        server.close()


def test_search_and_story_degrade_gracefully(mcp):
    """Whether or not /v1/search and the story view exist on this backend,
    both tools must return readable text, never a protocol error."""
    resp = mcp.rpc(30, "tools/call", {
        "name": "search_memory", "arguments": {"query": "Platform team"}})
    assert _tool_text(resp).strip()

    resp = mcp.rpc(31, "tools/call", {
        "name": "get_story", "arguments": {"name": "Nobody Nonexistent"}})
    text = _tool_text(resp)
    assert text.strip()
