"""POST /v1/remember — the natural-language ingestion front door.

Users type ANYTHING (or drop a document / a JSON payload). The system decides
what is temporal and deciphers the dates — nobody ever fills in a relation, an
entity picker, or a date field. ``occurred_at`` is optional and free-form
("last Tuesday" works); when absent or unparseable the entry is filed under now.

Every path funnels through the ONE write path, :func:`app.engine.ingest.ingest_text`
(Postgres episode → native Graphiti extraction/invalidation → entity projection).
The response stays HUMAN-FIRST: plain sentences under ``understood``
/ ``replaced`` / ``when_unclear``, machine ids under ``details``.

Graphiti remains the sole fact extractor. When extraction produces no graph
assertion, the episode ledger still retains the source and the response lists
it under ``remembered``. No parallel note store or second extraction path is
created. ``needs_review`` remains ``[]`` because there is no separate dispute queue.
``understood`` renders new graph edges, ``replaced`` renders Graphiti-native
invalidations, and ``when_unclear`` reports an unparseable free-form date.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.engine.ingest import IngestReport, flatten_json_text, ingest_text
from app.engine.native import format_validity
from app.normalize import normalize_time

router = APIRouter()

_UTC = timezone.utc
_GENRES = {"fiction", "nonfiction"}
_MAX_DOCUMENT_CHARS = 6000
_SENTENCE_BREAK_RE = re.compile(r"[.!?][\"'\u2019\u201d)\]]*\s+")


def _document_break(text: str, limit: int) -> int:
    """Choose a lossless structural boundary at or before ``limit``."""
    window = text[:limit]
    floor = limit // 2
    for start in (floor, 0):
        newline = window.rfind("\n", start)
        if newline >= 0:
            return newline + 1
        sentences = list(_SENTENCE_BREAK_RE.finditer(window, start))
        if sentences:
            return sentences[-1].end()
        for index in range(len(window) - 1, start - 1, -1):
            if window[index].isspace():
                return index + 1
    return limit


def chunk_document(content: str, max_chars: int = _MAX_DOCUMENT_CHARS) -> list[str]:
    """Split documents into bounded, ordered chunks without losing text."""
    remaining = content.strip()
    chunks: list[str] = []
    while len(remaining) > max_chars:
        cut = _document_break(remaining, max_chars)
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


class DocumentBody(BaseModel):
    content: str
    title: str | None = None


class ConversationMessage(BaseModel):
    """One actor turn, stored through Graphiti's native message extractor."""

    role: Literal["user", "assistant", "system"]
    content: str


class RememberIn(BaseModel):
    text: str | None = None
    json_payload: dict | None = Field(default=None, alias="json")
    document: DocumentBody | None = None
    conversation: list[ConversationMessage] | None = None
    occurred_at: str | None = None
    # Declared genre wins over any detection (pin > LLM).
    genre: str | None = None

    model_config = {"populate_by_name": True}


@router.post("/v1/remember")
def remember(body: RememberIn, request: Request) -> dict:
    ctx = request.state.tenant_ctx

    given = [x for x in (
        body.text, body.json_payload, body.document, body.conversation
    )
             if x is not None]
    if len(given) != 1:
        raise HTTPException(
            422, "Tell me one thing at a time: send exactly one of "
                 "text, json, document, or conversation.")
    if body.text is not None and not body.text.strip():
        raise HTTPException(422, "There's nothing to remember — the text is empty.")
    if body.conversation is not None:
        if not body.conversation:
            raise HTTPException(422, "There's nothing to remember — the conversation is empty.")
        if any(not message.content.strip() for message in body.conversation):
            raise HTTPException(422, "Conversation messages cannot be empty.")
    if body.genre is not None and body.genre not in _GENRES:
        raise HTTPException(422, f"invalid genre: {body.genre!r} "
                                 f"(expected one of {sorted(_GENRES)})")
    genre_meta = ({"genre": body.genre, "genre_source": "declared"}
                  if body.genre is not None else {})

    now = datetime.now(tz=_UTC)
    occurred_at = now
    when_unclear: list[str] = []
    if body.occurred_at and body.occurred_at.strip():
        resolved, unresolved = normalize_time(
            body.occurred_at, occurred_at=now,
            timezone=ctx.timezone, locale=ctx.locale)
        if unresolved:
            when_unclear.append(
                f"I couldn't make sense of “{body.occurred_at.strip()}” "
                f"as a date, so this is filed under today.")
        else:
            occurred_at = resolved

    # ---- run the one write path (one call per episode) ----------------------
    reports: list[IngestReport] = []
    note_candidates: list[tuple[IngestReport, str]] = []
    if body.text is not None:
        report = ingest_text(
            ctx, text=body.text.strip(), occurred_at=occurred_at,
            source="note", meta=genre_meta or None)
        reports.append(report)
        note_candidates.append((report, body.text.strip()))
    elif body.document is not None:
        chunks = chunk_document(body.document.content)
        if not chunks:
            raise HTTPException(
                422, "There's nothing to remember — the document is empty.")
        doc_id = str(uuid4())
        for ix, chunk in enumerate(chunks):
            meta = {"doc_id": doc_id, "chunk_ix": ix, **genre_meta}
            if body.document.title:
                meta["title"] = body.document.title
            report = ingest_text(
                ctx, text=chunk, occurred_at=occurred_at,
                source="doc", meta=meta)
            reports.append(report)
            note_candidates.append((report, chunk))
    elif body.json_payload is not None:
        flattened = flatten_json_text(body.json_payload)
        report = ingest_text(
            ctx, text=flattened, occurred_at=occurred_at,
            source="json_event", meta=genre_meta or None)
        reports.append(report)
        note_candidates.append((report, flattened))
    else:
        # Graphiti's message extractor is designed for one actor turn at a time.
        # Preserve native previous-message context with tiny ordering offsets
        # when the client supplies only a session-level timestamp.
        conversation_id = str(uuid4())
        for ix, message in enumerate(body.conversation or []):
            meta = {
                "conversation_id": conversation_id,
                "turn_index": ix,
                "role": message.role,
                **genre_meta,
            }
            rendered = f"{message.role}: {message.content.strip()}"
            report = ingest_text(
                ctx,
                text=rendered,
                occurred_at=occurred_at + timedelta(microseconds=ix),
                source="chat",
                meta=meta,
            )
            reports.append(report)
            note_candidates.append((report, rendered))

    remembered = [
        content
        for report, content in note_candidates
        if not report.deduped and not report.edges and not report.invalidated_edges
    ]

    return _render_response(reports, when_unclear, remembered)


# ---------------------------------------------------------------------------
# Human-first response rendering
# ---------------------------------------------------------------------------
def _closed_sentence(edge) -> str:
    """A closed edge rendered as a plain replacement sentence."""
    win = format_validity(edge)
    base = f"{edge.fact} ({win})" if win else edge.fact
    return f"{base} — this window was closed by newer information"


def _render_response(reports: list[IngestReport], when_unclear: list[str],
                     remembered: list[str] | None = None) -> dict:
    understood: list[str] = []
    replaced: list[str] = []
    fact_ids: list[str] = []
    seen_closed: set[str] = set()

    for report in reports:
        # understood: one sentence per NEW edge.
        for edge in report.edges:
            win = format_validity(edge)
            understood.append(f"{edge.fact} ({win})" if win else edge.fact)
            fact_ids.append(edge.uuid)

        # replaced: only Graphiti-native invalidations.
        for edge in report.invalidated_edges:
            if edge.uuid in seen_closed:
                continue
            seen_closed.add(edge.uuid)
            replaced.append(_closed_sentence(edge))

    return {
        "stored": True,
        "understood": understood,
        "remembered": remembered or [],
        "replaced": replaced,
        "needs_review": [],
        "when_unclear": when_unclear,
        "details": {
            "episode_ids": [r.episode_id for r in reports],
            "deduped": all(r.deduped for r in reports) if reports else False,
            "fact_ids": fact_ids,
            "time_unresolved": False,
            "time_unresolved_fact_ids": [],
        },
    }
