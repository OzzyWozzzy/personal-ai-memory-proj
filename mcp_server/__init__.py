"""Zero-dependency stdio MCP server for the temporal memory graph.

Hand-rolled JSON-RPC 2.0 over stdin/stdout (one JSON message per line, MCP
protocol version 2025-06-18) — Python standard library only, no MCP SDK, no
httpx. It is a thin bridge: every tool call is one REST request to the memory
backend, authenticated with ``MEMORY_API_KEY``.

Methods implemented: ``initialize``, ``notifications/initialized`` (no-op),
``tools/list``, ``tools/call``, ``ping``. Tool results are always
``[{"type": "text", "text": ...}]``; failures come back as tool results with
``isError`` — the server never crashes the agent's loop.

Environment:
    MEMORY_API_URL   backend base URL (default http://127.0.0.1:8321)
    MEMORY_API_KEY   an agent key from POST /v1/agent_keys (required)

Run: ``python -m mcp_server`` (see mcp_server/README.md for client setup).

"""
from __future__ import annotations

import json
import ipaddress
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {
    "name": "temporal-memory",
    "title": "Temporal Memory Graph",
    "version": "1.0.0",
}

TOOLS = [
    {
        "name": "remember",
        "title": "Remember in temporal memory",
        "description": (
            "Save something worth remembering — a fact, a decision, a change, "
            "or a note about a person or project. Include names and dates in "
            "the text when you know them. Use 'when' (a date like 2026-07-01) "
            "if the thing happened at a specific time."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string",
                         "description": "What to remember, in plain words."},
                "when": {"type": "string",
                         "description": "When it happened (ISO date or "
                                        "date-time), if known."},
            },
            "required": ["text"],
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    },
    {
        "name": "search_memory",
        "title": "Search temporal memory",
        "description": (
            "Search everything remembered — people, projects, decisions, "
            "documents, and meetings. Answer from the results conversationally, "
            "as things you already know; don't list every hit or cite where "
            "each one came from unless asked."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "What to look for."},
            },
            "required": ["query"],
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
    {
        "name": "get_context",
        "title": "Build temporal context",
        "description": (
            "Call before answering questions about people, projects, or "
            "history. Returns what's currently known as background for you — "
            "answer in natural, first-person prose as if you already remember "
            "it. Don't repeat the lines verbatim or recite dates and sources "
            "unless asked; just say what's true, and only flag a caveat when "
            "the memory genuinely conflicts or is out of date."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "The question or topic you need "
                                         "background on."},
                "max_words": {"type": "integer",
                              "description": "Rough size limit for the "
                                             "context (default about 1100 "
                                             "words)."},
            },
            "required": ["query"],
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
    {
        "name": "get_story",
        "title": "Get a temporal story",
        "description": (
            "Get how a person, project, or company has changed over time, as "
            "dated points from earliest to latest. Summarize the arc naturally: "
            "lead with what's true now and treat earlier states as history, "
            "not a data dump."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string",
                         "description": "Who or what to tell the story of."},
            },
            "required": ["name"],
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
]


class BackendError(Exception):
    """A REST failure with a human-readable message (returned as tool text)."""


_MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward an agent key to a redirect-selected destination."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _validated_base_url(value: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(value.rstrip("/"))
        port = parsed.port
    except ValueError as exc:
        raise BackendError("MEMORY_API_URL is not a valid backend URL.") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise BackendError(
            "MEMORY_API_URL must be an http(s) URL without credentials, "
            "a query, or a fragment."
        )
    if not (1 <= (port or (443 if parsed.scheme == "https" else 80)) <= 65535):
        raise BackendError("MEMORY_API_URL has an invalid port.")
    if parsed.scheme == "http":
        host = parsed.hostname.casefold()
        loopback = host == "localhost"
        if not loopback:
            try:
                loopback = ipaddress.ip_address(host).is_loopback
            except ValueError:
                loopback = False
        if not loopback:
            raise BackendError(
                "MEMORY_API_URL must use HTTPS unless the backend is loopback."
            )
    return urllib.parse.urlunsplit(parsed).rstrip("/")


class MemoryClient:
    """Minimal urllib REST client for the memory backend."""

    def __init__(self, base_url: str | None = None,
                 api_key: str | None = None):
        self.base = _validated_base_url(
            base_url or os.environ.get("MEMORY_API_URL")
            or "http://127.0.0.1:8321"
        )
        self.api_key = api_key if api_key is not None else \
            os.environ.get("MEMORY_API_KEY", "")
        proxy_mode = os.environ.get("MEMORY_API_TRUST_ENV_PROXY", "0").lower()
        if proxy_mode not in {"0", "1", "false", "true", "off", "on"}:
            raise BackendError("MEMORY_API_TRUST_ENV_PROXY must be a boolean.")
        handlers = [_NoRedirect()]
        if proxy_mode in {"0", "false", "off"}:
            handlers.insert(0, urllib.request.ProxyHandler({}))
        self._opener = urllib.request.build_opener(*handlers)

    def request(self, method: str, path: str, *, body: dict | None = None,
                params: dict | None = None) -> tuple[int, object]:
        if not self.api_key:
            raise BackendError(
                "MEMORY_API_KEY is not set — create a key with "
                "POST /v1/agent_keys and add it to this MCP server's "
                "configuration.")
        url = self.base + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={"X-API-Key": self.api_key,
                     "Content-Type": "application/json"})
        try:
            # URL scheme/host and redirect behavior are fixed above; the nosec
            # marker documents that this is not an arbitrary-scheme urlopen.
            with self._opener.open(req, timeout=30) as resp:  # nosec B310
                payload = resp.read(_MAX_RESPONSE_BYTES + 1)
                if len(payload) > _MAX_RESPONSE_BYTES:
                    raise BackendError("The memory backend response was too large.")
                raw = payload.decode("utf-8", "replace")
                return resp.status, (json.loads(raw) if raw else None)
        except urllib.error.HTTPError as exc:
            payload = exc.read(_MAX_RESPONSE_BYTES + 1)
            raw = payload[:_MAX_RESPONSE_BYTES].decode("utf-8", "replace")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"detail": raw[:500]}
            return exc.code, payload
        except urllib.error.URLError as exc:
            raise BackendError(
                f"Couldn't reach the memory backend at {self.base} — is it "
                f"running? ({exc.reason})")
        except TimeoutError:
            raise BackendError(
                f"The memory backend at {self.base} took too long to answer.")


# ---------------------------------------------------------------------------
# Response flattening: any backend JSON -> readable text.
# ---------------------------------------------------------------------------
_TEXT_KEYS = ("summary", "block", "message", "text", "line", "note")
_LIST_KEYS = ("lines", "summaries", "cards", "results", "items", "story",
              "chapters", "matches", "history", "memories", "connections")


def _flatten(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(s for s in (_flatten(v) for v in value) if s)
    if isinstance(value, dict):
        for key in _TEXT_KEYS:
            v = value.get(key)
            if isinstance(v, str) and v.strip():
                return v
        for key in _LIST_KEYS:
            v = value.get(key)
            if isinstance(v, list) and v:
                return _flatten(v)
        # A leaf card: join its short string fields into one line.
        strings = [str(v) for v in value.values()
                   if isinstance(v, (str, int, float)) and str(v).strip()]
        if strings and all(len(s) < 300 for s in strings):
            return " — ".join(strings)
        return json.dumps(value, indent=2, default=str)
    return str(value)


def _detail(payload) -> str:
    if isinstance(payload, dict) and payload.get("detail"):
        return str(payload["detail"])
    return _flatten(payload) or "no further detail"


def _find_id(value, keys=("subject_id", "entity_id")) -> str | None:
    """Depth-first search for the first plausible subject id in a response."""
    if isinstance(value, dict):
        for key in keys:
            v = value.get(key)
            if isinstance(v, str) and v:
                return v
        for v in value.values():
            found = _find_id(v, keys)
            if found:
                return found
    elif isinstance(value, list):
        for v in value:
            found = _find_id(v, keys)
            if found:
                return found
    return None


# ---------------------------------------------------------------------------
# Naturalize the /v1/context block: strip the fact:/source:/entity: prefixes,
# record ids, and "| sources:N" scaffolding so the caller reads plain
# sentences it can speak from, not a database dump. The statements, validity
# windows, and summaries (the actual content) are kept.
# ---------------------------------------------------------------------------
def _naturalize_line(line: str) -> str:
    kind, sep, rest = line.partition(":")
    if not sep or kind not in ("fact", "source", "entity", "community"):
        return line.strip()  # unknown line type: leave it untouched
    if kind == "fact":
        # fact:<id> | <statement> | <window> | sources:<n>
        body = rest.split(" | ", 1)[1] if " | " in rest else rest
        body = re.sub(r"\s*\|\s*sources:\d+\s*$", "", body)
        if " | " in body:
            statement, _, window = body.rpartition(" | ")
        else:
            statement, window = body, ""
        statement = statement.strip().rstrip(".")
        window = window.strip()
        if not statement:
            return ""
        if window and window.lower() != "timeless":
            return f"{statement} ({window})."
        return f"{statement}."
    if kind == "source":
        # source:<id> | <role> | <date> | <text> -> keep only <text>
        return rest.split(" | ", 3)[-1].strip()
    # entity/community:<name> | <summary>
    name, _, summary = rest.partition(" | ")
    name, summary = name.strip(), summary.strip()
    return f"{name} — {summary}" if summary else name


def _naturalize_block(block: str) -> str:
    naturalized = (_naturalize_line(ln) for ln in block.splitlines() if ln.strip())
    return "\n".join(ln for ln in naturalized if ln)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def tool_remember(client: MemoryClient, args: dict) -> str:
    text_ = (args.get("text") or "").strip()
    if not text_:
        return "Nothing to remember — 'text' was empty."
    body: dict = {"text": text_}
    if args.get("when"):
        body["occurred_at"] = args["when"]
    status, resp = client.request("POST", "/v1/remember", body=body)
    if status in (200, 201):
        if isinstance(resp, dict):
            lines: list[str] = []
            for key in ("understood", "replaced", "remembered",
                        "needs_review", "when_unclear"):
                v = resp.get(key)
                if isinstance(v, list):
                    lines += [s for s in v if isinstance(s, str) and s.strip()]
            if lines:
                return "\n".join(lines)
            if resp.get("stored"):
                return "Saved."
        return _flatten(resp) or "Remembered."
    return f"Couldn't save this: {_detail(resp)}"


def tool_search_memory(client: MemoryClient, args: dict) -> str:
    query = (args.get("query") or "").strip()
    if not query:
        return "Give me something to search for."
    status, resp = client.request("GET", "/v1/search",
                                  params={"q": query, "query": query})
    if status == 200:
        text_ = _render_search(resp) if isinstance(resp, dict) \
            else _flatten(resp)
        return text_ or f"Nothing remembered matches '{query}' yet."
    return f"Search failed: {_detail(resp)}"


def _render_search(resp: dict) -> str:
    """Flatten the /v1/search cards (matches + passages) to plain text."""
    parts: list[str] = []
    answer = resp.get("answer_block")
    if isinstance(answer, str) and answer.strip():
        parts.append(answer.strip())
    for m in resp.get("matches") or []:
        if not isinstance(m, dict):
            continue
        header = str(m.get("name") or "").strip()
        kind = str(m.get("kind") or "").strip()
        if header and kind:
            header = f"{header} ({kind})"
        if header:
            parts.append(header)
        for s in m.get("sentences") or []:
            if isinstance(s, str) and s.strip():
                parts.append(f"- {s.strip()}")
        for c in m.get("connections") or []:
            desc = c.get("description") if isinstance(c, dict) else None
            if isinstance(desc, str) and desc.strip():
                parts.append(f"- {desc.strip()}")
    for p in resp.get("passages") or []:
        if not isinstance(p, dict):
            continue
        line = str(p.get("text") or "").strip()
        if not line:
            continue
        # Provenance (when/source) is intentionally omitted: the caller should
        # speak from these as things it knows, not cite where each came from.
        parts.append(f"- {line}")
    if not parts:
        return _flatten(resp) if (resp.get("matches") or resp.get("passages")
                                  ) else ""
    return "\n".join(parts)


def tool_get_context(client: MemoryClient, args: dict) -> str:
    query = (args.get("query") or "").strip()
    if not query:
        return "Give me a question or topic to build context for."
    body: dict = {"query": query}
    max_words = args.get("max_words")
    if isinstance(max_words, (int, float)) and max_words > 0:
        body["token_budget"] = int(max_words * 1.3)
    else:
        body["token_budget"] = 1500
    status, resp = client.request("POST", "/v1/context", body=body)
    if status == 200:
        block = resp.get("block") if isinstance(resp, dict) else None
        text_ = _naturalize_block(block) \
            if isinstance(block, str) and block.strip() else _flatten(resp)
        return text_ or f"No stored context matches '{query}' yet."
    return f"Couldn't build context: {_detail(resp)}"


def tool_get_story(client: MemoryClient, args: dict) -> str:
    name = (args.get("name") or "").strip()
    if not name:
        return "Give me a person or project name."
    subject_id = None
    status, resp = client.request("GET", "/v1/search",
                                  params={"q": name, "query": name})
    if status == 200 and isinstance(resp, dict):
        matches = [m for m in (resp.get("matches") or [])
                   if isinstance(m, dict) and m.get("id")]
        with_story = [m for m in matches if m.get("story_available")]
        if with_story:
            subject_id = with_story[0]["id"]
        elif matches:
            subject_id = matches[0]["id"]
        else:
            subject_id = _find_id(resp)
    if subject_id is None:
        return (f"I couldn't find anyone or anything called '{name}' in "
                "memory yet.")
    status, resp = client.request("GET", f"/v1/subjects/{subject_id}/story")
    if status == 200:
        text_ = _render_story(resp) if isinstance(resp, dict) \
            else _flatten(resp)
        return text_ or f"No story recorded for '{name}' yet."
    return f"Couldn't fetch the story: {_detail(resp)}"


def _render_story(resp: dict) -> str:
    """Flatten /v1/subjects/{id}/story to dated lines, earliest to latest."""
    entries = resp.get("story")
    if not isinstance(entries, list) or not entries:
        return _flatten(resp) if entries else ""
    header = str(resp.get("name") or "").strip()
    kind = str(resp.get("kind") or "").strip()
    lines = [f"The story of {header} ({kind}):" if header and kind
             else (f"The story of {header}:" if header else "")]
    for e in entries:
        if not isinstance(e, dict):
            continue
        text_ = str(e.get("text") or "").strip()
        when = str(e.get("when") or "").strip()
        if text_:
            lines.append(f"{when} — {text_}" if when else text_)
    return "\n".join(ln for ln in lines if ln)


_TOOL_HANDLERS = {
    "remember": tool_remember,
    "search_memory": tool_search_memory,
    "get_context": tool_get_context,
    "get_story": tool_get_story,
}


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 server
# ---------------------------------------------------------------------------
class Server:
    def __init__(self, client: MemoryClient | None = None):
        self.client = client or MemoryClient()

    def handle_line(self, line: str) -> str | None:
        """One inbound message -> one outbound JSON string (or None for
        notifications). Never raises."""
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return json.dumps({"jsonrpc": "2.0", "id": None,
                               "error": {"code": -32700,
                                         "message": "Parse error"}})
        if not isinstance(msg, dict):
            return json.dumps({"jsonrpc": "2.0", "id": None,
                               "error": {"code": -32600,
                                         "message": "Invalid Request"}})
        method = msg.get("method")
        msg_id = msg.get("id")
        is_notification = "id" not in msg

        try:
            result = self._dispatch(method, msg.get("params") or {})
        except _RpcError as exc:
            if is_notification:
                return None
            return json.dumps({"jsonrpc": "2.0", "id": msg_id,
                               "error": {"code": exc.code,
                                         "message": exc.message}})
        except Exception as exc:  # absolute backstop: never crash the loop
            if is_notification:
                return None
            return json.dumps({"jsonrpc": "2.0", "id": msg_id,
                               "error": {"code": -32603,
                                         "message": f"Internal error: {exc}"}})
        if is_notification:
            return None
        return json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": result})

    def _dispatch(self, method: str | None, params: dict):
        if method == "initialize":
            requested = params.get("protocolVersion")
            version = requested if requested == PROTOCOL_VERSION \
                else PROTOCOL_VERSION
            return {"protocolVersion": version,
                    "capabilities": {"tools": {}},
                    "serverInfo": SERVER_INFO}
        if method in ("notifications/initialized", "initialized"):
            return {}  # no-op; notifications are silenced by the caller
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": TOOLS}
        if method == "tools/call":
            return self._call_tool(params)
        if method and method.startswith("notifications/"):
            return {}  # unknown notifications are ignored, never errors
        raise _RpcError(-32601, f"Method not found: {method}")

    def _call_tool(self, params: dict) -> dict:
        name = params.get("name")
        handler = _TOOL_HANDLERS.get(name)
        if handler is None:
            return _tool_error(f"Unknown tool: {name}. Available tools: "
                               + ", ".join(sorted(_TOOL_HANDLERS)))
        arguments = params.get("arguments") or {}
        try:
            text_ = handler(self.client, arguments)
        except BackendError as exc:
            return _tool_error(str(exc))
        except Exception as exc:  # tool errors are results, never crashes
            return _tool_error(f"Something went wrong: {exc}")
        return {"content": [{"type": "text", "text": text_}]}


def _tool_error(message: str) -> dict:
    return {"content": [{"type": "text", "text": message}], "isError": True}


class _RpcError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def main() -> int:
    if not os.environ.get("MEMORY_API_KEY"):
        print("mcp_server: warning: MEMORY_API_KEY is not set — tool calls "
              "will fail until it is configured.", file=sys.stderr)
    server = Server()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        out = server.handle_line(line)
        if out is not None:
            sys.stdout.write(out + "\n")
            sys.stdout.flush()
    return 0
