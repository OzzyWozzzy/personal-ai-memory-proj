"""Serve the local graph and governance inspector at ``/ui``.

The frontend is plain static files under ``frontend/`` (no build step). This
The shell is public, while every data request uses the HttpOnly local-session
cookie. Assets use an explicit path-safe route and require no build step.
"""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

ui_router = APIRouter(tags=["ui"])

_FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".json": "application/json",
}


def _serve(filename: str) -> FileResponse:
    # Reject anything that is not a bare, safe filename (no path traversal).
    if not _SAFE_NAME.match(filename) or filename.startswith("."):
        raise HTTPException(404, "not found")
    path = (_FRONTEND_DIR / filename).resolve()
    if _FRONTEND_DIR.resolve() not in path.parents or not path.is_file():
        raise HTTPException(404, "not found")
    media_type = _CONTENT_TYPES.get(path.suffix, "application/octet-stream")
    # The app iterates quickly and ships no cache-busting hashes; a stale
    # cached app.js against a newer backend is worse than the extra fetch.
    return FileResponse(path, media_type=media_type,
                        headers={"Cache-Control": "no-store"})


@ui_router.get("/ui")
def ui_index() -> FileResponse:
    return _serve("index.html")


@ui_router.get("/ui/{filename}")
def ui_asset(filename: str) -> FileResponse:
    return _serve(filename)
