"""Small presentation helpers for Graphiti's native temporal fields.

Graphiti owns ``valid_at``/``invalid_at`` and all mutations of those fields.
This module deliberately does not infer missing dates, repair windows, choose a
"latest" value, or perform Allen-algebra supersession.  It only normalizes
Graphiti values for JSON/display and evaluates a conventional as-of view over
the dates Graphiti actually stored.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

_UTC = timezone.utc


def value(obj, name: str, default=None):
    """Read ``name`` from a Graphiti model or a mapping."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def aware(dt: datetime | None) -> datetime | None:
    """Return a UTC-aware datetime without inventing a missing value."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=_UTC)
    return dt.astimezone(_UTC)


def validity(edge) -> tuple[datetime | None, datetime | None]:
    """Return only Graphiti's asserted validity bounds for ``edge``."""
    return aware(value(edge, "valid_at")), aware(value(edge, "invalid_at"))


def format_validity(edge) -> str:
    """Compact display of Graphiti's actual validity fields."""
    start, end = validity(edge)
    if start is None and end is None:
        return ""
    start_s = start.date().isoformat() if start is not None else "unknown"
    if end is None:
        return f"since {start_s}" if start is not None else "until present"
    end_s = end.date().isoformat()
    if start is not None and start == end:
        return f"on {start_s}"
    return f"{start_s} to {end_s}"


def is_visible_at(edge, at: datetime) -> bool:
    """Whether Graphiti's stored validity contains ``at``.

    An edge with no native validity dates is timeless and therefore visible.
    Bounds are start-inclusive and end-exclusive, matching Graphiti's search
    helper prompt and common valid-time semantics.
    """
    at = aware(at)
    if at is None:  # defensive; callers provide a real instant
        return True
    start, end = validity(edge)
    if start is not None and start > at:
        return False
    if end is not None and end <= at:
        return False
    return True


def filter_visible(edges, at: datetime, *, include_historical: bool = False):
    """Filter by Graphiti validity, optionally retaining already-started history."""
    at = aware(at)
    if include_historical:
        return [edge for edge in edges if validity(edge)[0] in (None,) or validity(edge)[0] <= at]
    return [edge for edge in edges if is_visible_at(edge, at)]


def normalize_relation(name: str | None) -> str:
    """Normalize a relation label for display filtering only."""
    return re.sub(r"[^a-z0-9]+", "_", (name or "").casefold()).strip("_")
