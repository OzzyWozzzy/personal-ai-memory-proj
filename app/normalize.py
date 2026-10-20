"""Deterministic time normalization and object canonicalization.

Pure functions only: no DB, no network, no wall-clock reads, no LLM. Naive
datetimes (whether ``raw`` or ``occurred_at``) are interpreted in the tenant
timezone; everything is returned as a UTC-aware datetime.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from uuid import UUID

import dateparser
from zoneinfo import ZoneInfo

_UTC = ZoneInfo("UTC")

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_QUARTER_LABELED = re.compile(r"q([1-4])\s+(\d{4})|(\d{4})\s+q([1-4])")
_QUARTER_RELATIVE = re.compile(
    r"(this|current|next|last|previous|coming|prior)\s+quarter"
)
_QUARTER_OFFSET = {
    "this": 0, "current": 0, "next": 1, "coming": 1,
    "last": -1, "previous": -1, "prior": -1,
}
_WEEKDAY_PHRASE = re.compile(
    r"(last|next|this|previous|coming)\s+"
    r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
)


# ---------------------------------------------------------------------------
# time normalization
# ---------------------------------------------------------------------------
def normalize_time(
    raw: str | datetime | None,
    *,
    occurred_at: datetime,
    timezone: str = "UTC",
    locale: str = "en",
) -> tuple[datetime, bool]:
    """Resolve ``raw`` to ``(valid_from UTC-aware, time_unresolved)``.

    Deterministic cascade (occurred_at + tenant tz/locale are the only context):

    1. Verbatim system-of-record copy is handled by CALLERS — not special-cased here.
    2. ``raw`` is a ``datetime`` → converted to UTC (naive read in tenant tz).
    3. ``raw`` is ``None`` → ``(occurred_at→UTC, False)`` (present-tense: true
       as of the episode).
    4. ``raw`` is a ``str`` → explicit ISO first, then deterministic quarter and
       weekday pre-rules, then ``dateparser`` anchored to ``occurred_at``.
    5. Anything unresolvable → ``(occurred_at→UTC, True)``. Never guess; never
       read the current clock.
    """
    occ_utc = _to_utc(occurred_at, timezone)

    if isinstance(raw, datetime):
        return _to_utc(raw, timezone), False
    if raw is None:
        return occ_utc, False

    text = raw.strip()
    if text:
        base_local = occ_utc.astimezone(ZoneInfo(timezone)).replace(tzinfo=None)
        iso = _try_iso(text, timezone)
        if iso is not None:
            return iso, False
        quarter = _try_quarter(text, base_local)
        if quarter is not None:
            return _local_to_utc(quarter, timezone), False
        weekday = _try_weekday(text, base_local)
        if weekday is not None:
            return _local_to_utc(weekday, timezone), False
        parsed = _try_dateparser(text, base_local, timezone, locale)
        if parsed is not None:
            return parsed.astimezone(_UTC), False

    return occ_utc, True


def _to_utc(dt: datetime, tz: str) -> datetime:
    """Naive → interpreted in tenant tz; aware → converted. Result is UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(tz))
    return dt.astimezone(_UTC)


def _local_to_utc(naive_local: datetime, tz: str) -> datetime:
    return naive_local.replace(tzinfo=ZoneInfo(tz)).astimezone(_UTC)


def _try_iso(text: str, tz: str) -> datetime | None:
    """Unambiguous ISO-8601 absolute forms. Date-only → tenant-tz midnight."""
    try:
        d = date.fromisoformat(text)
        return datetime(d.year, d.month, d.day, tzinfo=ZoneInfo(tz)).astimezone(_UTC)
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return _to_utc(dt, tz)


def _try_quarter(text: str, base_local: datetime) -> datetime | None:
    """Quarter expressions dateparser cannot handle. Returns naive local start."""
    t = text.casefold().strip()
    m = _QUARTER_LABELED.fullmatch(t)
    if m:
        if m.group(1):          # "Q3 2026"
            quarter, year = int(m.group(1)), int(m.group(2))
        else:                   # "2026 Q4"
            year, quarter = int(m.group(3)), int(m.group(4))
        return _quarter_start(year, quarter)
    m = _QUARTER_RELATIVE.fullmatch(t)
    if m:
        cur = (base_local.month - 1) // 3 + 1
        index = base_local.year * 4 + (cur - 1) + _QUARTER_OFFSET[m.group(1)]
        return _quarter_start(index // 4, index % 4 + 1)
    return None


def _quarter_start(year: int, quarter: int) -> datetime:
    return datetime(year, (quarter - 1) * 3 + 1, 1)


def _try_weekday(text: str, base_local: datetime) -> datetime | None:
    """"last/next/this <weekday>" — midnight of the resolved day (tenant local).

    dateparser 1.4.1 resolves a bare weekday but returns None for these
    modifier phrases, so we compute them deterministically from ``occurred_at``.
    """
    m = _WEEKDAY_PHRASE.fullmatch(text.casefold().strip())
    if not m:
        return None
    modifier, target = m.group(1), _WEEKDAYS[m.group(2)]
    base_wd = base_local.weekday()
    if modifier in ("last", "previous"):
        delta = (base_wd - target) % 7 or 7
        day = base_local - timedelta(days=delta)
    elif modifier in ("next", "coming"):
        delta = (target - base_wd) % 7 or 7
        day = base_local + timedelta(days=delta)
    else:  # "this"
        day = base_local + timedelta(days=(target - base_wd) % 7)
    return datetime(day.year, day.month, day.day)


def _try_dateparser(
    text: str, base_local: datetime, tz: str, locale: str
) -> datetime | None:
    """dateparser anchored to occurred_at. Settings chosen empirically — see
    the module tests and the final report for the rationale (notably why
    STRICT_PARSING is NOT used: it rejects "two weeks ago" and other required
    relative expressions)."""
    lang = re.split(r"[-_]", locale)[0].lower() or "en"
    return dateparser.parse(
        text,
        languages=[lang],
        settings={
            "RELATIVE_BASE": base_local,
            "TIMEZONE": tz,
            "RETURN_AS_TIMEZONE_AWARE": True,
            "TO_TIMEZONE": "UTC",
        },
    )


# ---------------------------------------------------------------------------
# object canonicalization
# ---------------------------------------------------------------------------
def compute_object_norm(
    *,
    object_value: str | None,
    object_entity: UUID | None,
    object_type: str,
) -> str:
    """Deterministic canonical key-equality form for a fact's object.

    entity → ``str(UUID)``; number → canonical decimal (no exponent, no trailing
    zeros); date → ISO (``YYYY-MM-DD`` for dates, UTC ISO-8601 for datetimes);
    text / ``enum:*`` → strip, casefold, collapse internal whitespace.

    Exactly one of ``object_value`` / ``object_entity`` must be provided.
    """
    if (object_value is None) == (object_entity is None):
        raise ValueError(
            "exactly one of object_value / object_entity must be provided "
            f"(got object_value={object_value!r}, object_entity={object_entity!r})"
        )

    if object_type == "entity":
        if object_entity is None:
            raise ValueError("object_type 'entity' requires object_entity")
        return str(object_entity)

    if object_value is None:
        raise ValueError(f"object_type {object_type!r} requires object_value")

    if object_type == "number":
        return _norm_number(object_value)
    if object_type == "date":
        return _norm_date(object_value)
    return _collapse(object_value)


def _norm_number(value: str) -> str:
    try:
        d = Decimal(value.strip()).normalize()
    except InvalidOperation as exc:
        raise ValueError(f"not a valid number: {value!r}") from exc
    if d == 0:                      # collapse -0 / 0E+n to canonical '0'
        d = Decimal(0)
    return format(d, "f")


def _norm_date(value: str) -> str:
    text = value.strip()
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"not a valid date/datetime: {value!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_UTC)
    return dt.astimezone(_UTC).isoformat()


def _collapse(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()
