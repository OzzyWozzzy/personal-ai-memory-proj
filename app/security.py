"""Small HTTP boundary for the local API and bundled graph inspector."""
from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlsplit

from starlette.responses import JSONResponse, Response

DEV_PREVIEW_COOKIE = "memory-preview-key"
_UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
_CORS_METHODS = "GET, HEAD, POST, PUT, PATCH, DELETE, OPTIONS"
_CORS_HEADERS = {"accept", "content-type", "x-api-key"}


def _boolean(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean")


def _csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _normal_origin(raw: str) -> str:
    try:
        parsed = urlsplit(raw.strip())
        port = parsed.port
    except ValueError as exc:
        raise RuntimeError("allowed origins must be bare http(s) origins") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("allowed origins must be bare http(s) origins")
    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    default_port = 80 if parsed.scheme == "http" else 443
    suffix = f":{port}" if port and port != default_port else ""
    return f"{parsed.scheme.lower()}://{host}{suffix}"


def local_preview_enabled() -> bool:
    """The UI is local-first and enabled by default; it can be disabled."""
    return _boolean("MEMORY_ENABLE_LOCAL_PREVIEW", True)


@dataclass(frozen=True)
class SecurityConfig:
    allowed_hosts: tuple[str, ...]
    allowed_origins: frozenset[str]
    max_request_bytes: int

    @classmethod
    def from_env(cls) -> "SecurityConfig":
        hosts = _csv(
            "MEMORY_ALLOWED_HOSTS", ("127.0.0.1", "localhost", "testserver", "::1")
        )
        if not hosts or any(host == "*" for host in hosts):
            raise RuntimeError("MEMORY_ALLOWED_HOSTS must list explicit hosts")
        origins = frozenset(
            _normal_origin(value)
            for value in _csv(
                "MEMORY_ALLOWED_ORIGINS",
                (
                    "http://127.0.0.1:8321",
                    "http://localhost:8321",
                    "http://testserver",
                ),
            )
        )
        try:
            maximum = int(os.environ.get("MEMORY_MAX_REQUEST_BYTES", 2 * 1024 * 1024))
        except ValueError as exc:
            raise RuntimeError("MEMORY_MAX_REQUEST_BYTES must be an integer") from exc
        if maximum <= 0:
            raise RuntimeError("MEMORY_MAX_REQUEST_BYTES must be positive")
        return cls(hosts, origins, maximum)


def _headers(scope) -> dict[bytes, bytes]:
    return {key.lower(): value for key, value in scope.get("headers", [])}


def _header_text(headers: dict[bytes, bytes], name: bytes) -> str:
    return headers.get(name, b"").decode("latin-1").strip()


def _host_name(value: str) -> str:
    try:
        return (urlsplit("//" + value).hostname or "").lower()
    except ValueError:
        return ""


def _set_header(headers: list[tuple[bytes, bytes]], name: str, value: str) -> None:
    key = name.lower().encode("latin-1")
    headers[:] = [(k, v) for k, v in headers if k.lower() != key]
    headers.append((key, value.encode("latin-1")))


class SecurityMiddleware:
    """Enforce the local host/origin boundary and attach browser headers."""

    def __init__(self, app, config: SecurityConfig):
        self.app = app
        self.config = config

    def _response_headers(self, origin: str | None, path: str) -> dict[str, str]:
        out = {
            "Content-Security-Policy": (
                "default-src 'self'; base-uri 'self'; frame-ancestors 'none'; "
                "object-src 'none'; script-src 'self'; "
                "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                "connect-src 'self'; form-action 'self'"
            ),
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Referrer-Policy": "no-referrer",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Resource-Policy": "same-origin",
        }
        if path.startswith("/v1/"):
            out["Cache-Control"] = "no-store"
        if origin is not None:
            out.update(
                {
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Credentials": "true",
                    "Vary": "Origin",
                }
            )
        return out

    async def _reject(self, scope, receive, send, status: int, detail: str) -> None:
        response = JSONResponse({"detail": detail}, status_code=status)
        await response(scope, receive, send)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = _headers(scope)
        host = _host_name(_header_text(headers, b"host"))
        if host not in {item.strip("[]").lower() for item in self.config.allowed_hosts}:
            await self._reject(scope, receive, send, 400, "Invalid Host header")
            return

        raw_origin = _header_text(headers, b"origin")
        origin = None
        if raw_origin:
            try:
                origin = _normal_origin(raw_origin)
            except RuntimeError:
                await self._reject(scope, receive, send, 403, "Origin not allowed")
                return
            if origin not in self.config.allowed_origins:
                await self._reject(scope, receive, send, 403, "Origin not allowed")
                return

        method = scope.get("method", "GET").upper()
        cookie = _header_text(headers, b"cookie")
        if method in _UNSAFE_METHODS and DEV_PREVIEW_COOKIE in cookie and origin is None:
            await self._reject(scope, receive, send, 403, "Origin required")
            return

        if method == "OPTIONS" and _header_text(
            headers, b"access-control-request-method"
        ):
            if origin is None:
                await self._reject(scope, receive, send, 403, "Origin not allowed")
                return
            requested = {
                item.strip().lower()
                for item in _header_text(
                    headers, b"access-control-request-headers"
                ).split(",")
                if item.strip()
            }
            if not requested.issubset(_CORS_HEADERS):
                await self._reject(scope, receive, send, 403, "CORS headers not allowed")
                return
            response = Response(
                status_code=204,
                headers={
                    **self._response_headers(origin, scope.get("path", "")),
                    "Access-Control-Allow-Methods": _CORS_METHODS,
                    "Access-Control-Allow-Headers": ", ".join(sorted(_CORS_HEADERS)),
                },
            )
            await response(scope, receive, send)
            return

        async def secured_send(message):
            if message["type"] == "http.response.start":
                response_headers = list(message.get("headers", []))
                for name, value in self._response_headers(
                    origin, scope.get("path", "")
                ).items():
                    _set_header(response_headers, name, value)
                message["headers"] = response_headers
            await send(message)

        await self.app(scope, receive, secured_send)


class RequestBodyLimitMiddleware:
    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        raw = _header_text(_headers(scope), b"content-length")
        if raw:
            try:
                too_large = int(raw) > self.max_bytes
            except ValueError:
                await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(
                    scope, receive, send
                )
                return
            if too_large:
                await JSONResponse({"detail": "Request body too large"}, status_code=413)(
                    scope, receive, send
                )
                return
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise ValueError("request body too large")
            return message

        try:
            await self.app(scope, limited_receive, send)
        except ValueError as exc:
            if str(exc) != "request body too large":
                raise
            await JSONResponse({"detail": "Request body too large"}, status_code=413)(
                scope, receive, send
            )


def install_security_middleware(app, config: SecurityConfig) -> None:
    app.add_middleware(RequestBodyLimitMiddleware, max_bytes=config.max_request_bytes)
    app.add_middleware(SecurityMiddleware, config=config)
