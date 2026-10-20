"""Table-driven tests for deterministic time + object normalization.

Runs with NO database and NO network. All relative-date resolution is anchored
to a fixed ``occurred_at`` (never the wall clock), so results are deterministic.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import pytest

from app.normalize import compute_object_norm, normalize_time

UTC = timezone.utc
# A mid-week anchor: Wednesday 2026-07-15 14:00 UTC == 10:00 in America/New_York.
OCC = datetime(2026, 7, 15, 14, 0, tzinfo=UTC)
NY = "America/New_York"


# (id, raw, occurred_at, tz, locale, expected_dt, expected_unresolved)
TIME_CASES = [
    (
        "iso_datetime_utc",
        "2026-03-15T09:30:00+00:00", OCC, NY, "en",
        datetime(2026, 3, 15, 9, 30, tzinfo=UTC), False,
    ),
    (
        "iso_datetime_z",
        "2026-03-15T09:30:00Z", OCC, NY, "en",
        datetime(2026, 3, 15, 9, 30, tzinfo=UTC), False,
    ),
    (
        "iso_datetime_offset",  # -05:00 → +5h to UTC
        "2026-03-15T09:30:00-05:00", OCC, NY, "en",
        datetime(2026, 3, 15, 14, 30, tzinfo=UTC), False,
    ),
    (
        "iso_date_midnight_tenant_tz",  # NY midnight on Mar 15 (EDT, UTC-4)
        "2026-03-15", OCC, NY, "en",
        datetime(2026, 3, 15, 4, 0, tzinfo=UTC), False,
    ),
    (
        "locale_formatted_date_de",  # German "15. März 2026", NY midnight
        "15. März 2026", OCC, NY, "de",
        datetime(2026, 3, 15, 4, 0, tzinfo=UTC), False,
    ),
    (
        "last_friday_from_wednesday",  # most recent past Friday == Jul 10
        "last Friday", OCC, NY, "en",
        datetime(2026, 7, 10, 4, 0, tzinfo=UTC), False,
    ),
    (
        "two_weeks_ago",  # base 2026-07-15 14:00Z − 14d
        "two weeks ago", OCC, NY, "en",
        datetime(2026, 7, 1, 14, 0, tzinfo=UTC), False,
    ),
    (
        "next_quarter",  # base in Q3 → Q4 start, Oct 1 NY midnight (EDT)
        "next quarter", OCC, NY, "en",
        datetime(2026, 10, 1, 4, 0, tzinfo=UTC), False,
    ),
    (
        "q3_2026",  # labeled quarter, Jul 1 NY midnight (EDT)
        "Q3 2026", OCC, NY, "en",
        datetime(2026, 7, 1, 4, 0, tzinfo=UTC), False,
    ),
    (
        "q1_2026",  # Jan 1 NY midnight (EST, UTC-5)
        "Q1 2026", OCC, NY, "en",
        datetime(2026, 1, 1, 5, 0, tzinfo=UTC), False,
    ),
    (
        "unresolvable_text",  # never guess → fall back to occurred_at, flagged
        "when the stars align", OCC, NY, "en",
        OCC.astimezone(UTC), True,
    ),
    (
        "none_passthrough",  # present-tense: true as of the episode
        None, OCC, NY, "en",
        OCC.astimezone(UTC), False,
    ),
    (
        "naive_datetime_in_tenant_tz",  # 09:30 EDT (UTC-4) → 13:30 UTC
        datetime(2026, 3, 15, 9, 30), OCC, NY, "en",
        datetime(2026, 3, 15, 13, 30, tzinfo=UTC), False,
    ),
    (
        "aware_datetime_passthrough",  # already aware → convert to UTC
        datetime(2026, 3, 15, 9, 30, tzinfo=UTC), OCC, NY, "en",
        datetime(2026, 3, 15, 9, 30, tzinfo=UTC), False,
    ),
]


@pytest.mark.parametrize(
    "raw,occurred_at,tz,locale,expected_dt,expected_unresolved",
    [c[1:] for c in TIME_CASES],
    ids=[c[0] for c in TIME_CASES],
)
def test_normalize_time(raw, occurred_at, tz, locale, expected_dt, expected_unresolved):
    got, unresolved = normalize_time(
        raw, occurred_at=occurred_at, timezone=tz, locale=locale
    )
    assert got == expected_dt
    assert got.utcoffset().total_seconds() == 0  # returned aware in UTC
    assert unresolved is expected_unresolved


def test_normalize_time_repeat_call_is_byte_equal():
    """Determinism: identical inputs twice → byte-equal outputs, every case."""
    for _id, raw, occurred_at, tz, locale, *_ in TIME_CASES:
        first = normalize_time(raw, occurred_at=occurred_at, timezone=tz, locale=locale)
        second = normalize_time(raw, occurred_at=occurred_at, timezone=tz, locale=locale)
        assert first[0].isoformat() == second[0].isoformat()
        assert first[1] is second[1]


def test_normalize_time_never_reads_wall_clock():
    """A relative expression resolves purely from occurred_at: changing the
    anchor by exactly one day shifts the answer by exactly one day."""
    other = datetime(2026, 7, 16, 14, 0, tzinfo=UTC)  # Thursday
    a, _ = normalize_time("two weeks ago", occurred_at=OCC, timezone=NY)
    b, _ = normalize_time("two weeks ago", occurred_at=other, timezone=NY)
    assert (b - a).days == 1


# ---------------------------------------------------------------------------
# object_norm
# ---------------------------------------------------------------------------
_ENT = UUID("12345678-1234-5678-1234-567812345678")

# (id, object_value, object_entity, object_type, expected)
OBJECT_CASES = [
    ("entity_uuid", None, _ENT, "entity", str(_ENT)),
    ("number_trailing_zeros", "50.00", None, "number", "50"),
    ("number_leading_fraction", "0.500", None, "number", "0.5"),
    ("number_scientific", "1e3", None, "number", "1000"),
    ("number_plain_int", "1000", None, "number", "1000"),
    ("number_negative", "-50.00", None, "number", "-50"),
    ("number_zero", "0.0", None, "number", "0"),
    ("number_negative_zero", "-0.0", None, "number", "0"),
    ("number_whitespace", "  42  ", None, "number", "42"),
    ("number_small", "1E-3", None, "number", "0.001"),
    ("date_iso", "2026-03-15", None, "date", "2026-03-15"),
    ("date_datetime_utc", "2026-03-15T09:30:00+00:00", None, "date", "2026-03-15T09:30:00+00:00"),
    ("date_datetime_offset", "2026-03-15T09:30:00-05:00", None, "date", "2026-03-15T14:30:00+00:00"),
    ("date_datetime_naive_utc", "2026-03-15T09:30:00", None, "date", "2026-03-15T09:30:00+00:00"),
    ("text_case_and_whitespace", "  Hello   World  ", None, "text", "hello world"),
    ("text_upper", "ACME", None, "text", "acme"),
    ("text_unicode_case", "STRASSE", None, "text", "strasse"),
    ("enum_casefold", "Active", None, "enum:status", "active"),
    ("enum_whitespace", "  In   Progress ", None, "enum:status", "in progress"),
]


@pytest.mark.parametrize(
    "object_value,object_entity,object_type,expected",
    [c[1:] for c in OBJECT_CASES],
    ids=[c[0] for c in OBJECT_CASES],
)
def test_compute_object_norm(object_value, object_entity, object_type, expected):
    assert compute_object_norm(
        object_value=object_value,
        object_entity=object_entity,
        object_type=object_type,
    ) == expected


def test_compute_object_norm_requires_exactly_one():
    with pytest.raises(ValueError):
        compute_object_norm(object_value=None, object_entity=None, object_type="text")
    with pytest.raises(ValueError):
        compute_object_norm(
            object_value="x", object_entity=_ENT, object_type="text"
        )


def test_compute_object_norm_rejects_bad_number():
    with pytest.raises(ValueError):
        compute_object_norm(object_value="not-a-number", object_entity=None, object_type="number")
