"""Plain-language rendering of ledger rows (the natural-language UX layer).

Everything the /v1/remember, /v1/search and /v1/subjects/{id}/story endpoints
show a human goes through this module, so the vocabulary lives in ONE place and
the frontend can stay dumb: no "entity type", no "relation", no "valid_from" —
just sentences like "Jane lives in Kaohsiung (since May 2, 2026)".

Pure functions only: no DB, no network, no LLM. Callers pass the fact rows the
temporal predicates in :mod:`app.queries` already produced plus an id → name
map; nothing here re-decides temporal truth.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

_UTC = timezone.utc

# ---------------------------------------------------------------------------
# Relation templates: relation -> (present, past, future) verb phrases.
# The subject goes before the phrase, the object after it. Events use only the
# first slot (their verb is already a past form: "shipped").
# ---------------------------------------------------------------------------
_RELATION_FORMS: dict[str, tuple[str, str, str]] = {
    "lives_in": ("lives in", "used to live in", "will live in"),
    "has_role": ("is", "was", "will be"),
    "works_at": ("works at", "used to work at", "will work at"),
    "primary_contact_of": ("is the main contact for",
                           "used to be the main contact for",
                           "will be the main contact for"),
    "owner_of_account": ("owns the account", "used to own the account",
                         "will own the account"),
    "reports_to": ("reports to", "used to report to", "will report to"),
    "has_skill": ("knows", "used to know", "will know"),
    "uses_vendor": ("uses", "used to use", "will use"),
    "member_of": ("is part of", "used to be part of", "will be part of"),
    "order_shipped": ("shipped", "shipped", "shipped"),
    "meeting_held": ("met about", "met about", "met about"),
    "invoice_disputed": ("disputed an invoice", "disputed an invoice",
                         "disputed an invoice"),
    "defined_as": ("means", "meant", "will mean"),
    "part_of": ("is part of", "was part of", "will be part of"),
    "is_a": ("is a", "was a", "will be a"),
}

_TYPE_LABELS = {
    "person": "Person",
    "org": "Company",
    "account": "Customer account",
    "policy": "Policy",
    "concept": "Idea / term",
    "category": "Idea / term",
    "product": "Product",
    "thing": "Thing",
}

_SOURCE_LABELS = {
    "note": "note",
    "chat": "chat",
    "email": "email",
    "doc": "document",
    "manual_fact": "manual entry",
    "json_event": "data record",
}

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

# Relations stored account-subject but read person-first (see humanize_fact).
_OBJECT_FIRST_RENDER = {"primary_contact_of", "owner_of_account"}


# ---------------------------------------------------------------------------
# Small public helpers
# ---------------------------------------------------------------------------
def humanize_relation(relation: str) -> str:
    """The present-tense plain phrase for a relation ('lives_in' -> 'lives in').
    Unknown relations fall back to their words ('prefers_channel' -> 'prefers
    channel') — readable even before an admin reviews them."""
    forms = _RELATION_FORMS.get(relation)
    if forms is not None:
        return forms[0]
    return relation.replace("_", " ")


def relation_forms(relation: str) -> tuple[str, str, str]:
    """(present, past, future) phrases with a generic fallback for novel
    relations: 'previously <words>' / '<words> will be'."""
    forms = _RELATION_FORMS.get(relation)
    if forms is not None:
        return forms
    words = relation.replace("_", " ")
    return (words, f"previously {words}", f"{words} will be")


def humanize_type(entity_type: str | None) -> str:
    """'person' -> 'Person', 'org' -> 'Company', ... fallback Title Case."""
    if not entity_type:
        return "Thing"
    return _TYPE_LABELS.get(entity_type, entity_type.replace("_", " ").title())


def humanize_source(source: str | None) -> str:
    """Episode source -> plain word for "From a {source} on {date}" lines."""
    if not source:
        return "note"
    return _SOURCE_LABELS.get(source, source.replace("_", " "))


def friendly_date(dt: datetime | None) -> str:
    """'May 2, 2026' — never ISO. Naive datetimes are treated as UTC."""
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_UTC)
    dt = dt.astimezone(_UTC)
    return f"{_MONTHS[dt.month - 1]} {dt.day}, {dt.year}"


def _friendly_range(a: datetime, b: datetime) -> str:
    """'Jan 10 – May 2, 2026' (first year elided when both match)."""
    a = a.astimezone(_UTC) if a.tzinfo else a.replace(tzinfo=_UTC)
    b = b.astimezone(_UTC) if b.tzinfo else b.replace(tzinfo=_UTC)
    if a.year == b.year:
        first = f"{_MONTHS[a.month - 1]} {a.day}"
    else:
        first = friendly_date(a)
    return f"{first} – {friendly_date(b)}"


def _sentence(s: str) -> str:
    s = s.strip()
    if s and not s[0].isupper():
        s = s[0].upper() + s[1:]
    return s


def _name_of(entity_names: dict, key) -> str | None:
    if key is None:
        return None
    if key in entity_names:
        return entity_names[key]
    return entity_names.get(str(key), str(key))


# ---------------------------------------------------------------------------
# The fact renderer
# ---------------------------------------------------------------------------
def humanize_fact(fact_row: dict, entity_names: dict, *,
                  now: datetime | None = None) -> str:
    """One fact row -> one tense-aware plain sentence.

    * open state, past start ..... "Jane lives in Kaohsiung (since May 2, 2026)"
    * closed state ............... "Jane used to live in Taipei (Jan 10 – May 2, 2026)"
    * future-dated open state .... "From Jul 15, 2026, expense policy reimbursement
                                    limit will be 75"
    * event ...................... "On Mar 3, 2026, Acme shipped order o-1"
    * atemporal .................. "Churn means customer cancellation" (never dated)
    * disputed ................... trailing "(unconfirmed — conflicting information)"

    ``fact_row`` is a row dict from :mod:`app.queries` (``effective_valid_to``
    wins over ``valid_to`` when present, so as-believed views render what was
    believed). ``entity_names`` maps entity id (UUID or str) -> display name.
    """
    subj = _name_of(entity_names, fact_row["subject_id"]) or "someone"
    if fact_row.get("object_entity") is not None:
        obj = _name_of(entity_names, fact_row["object_entity"])
    else:
        obj = str(fact_row.get("object_value") or "")

    relation = fact_row["relation"]
    # Object-first relations: the ledger keys these on the ACCOUNT (an account
    # has exactly one contact/owner — that's what single-valued supersession
    # protects), but the sentence reads person-first. Swap for display only.
    if relation in _OBJECT_FIRST_RENDER:
        subj, obj = obj, subj
    present, past, future = relation_forms(relation)
    tclass = fact_row.get("temporal_class") or "state"
    disputed = bool(fact_row.get("disputed"))

    if now is None:
        now = datetime.now(tz=_UTC)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=_UTC)

    valid_from = fact_row.get("valid_from")
    if valid_from is not None and valid_from.tzinfo is None:
        valid_from = valid_from.replace(tzinfo=_UTC)
    if "effective_valid_to" in fact_row:
        valid_to = fact_row["effective_valid_to"]
    else:
        valid_to = fact_row.get("valid_to")
    if valid_to is not None and valid_to.tzinfo is None:
        valid_to = valid_to.replace(tzinfo=_UTC)

    if tclass == "atemporal":
        text = f"{subj} {present} {obj}"
    elif tclass == "event":
        text = f"On {friendly_date(valid_from)}, {subj} {present} {obj}"
    elif valid_to is not None:
        if valid_to == valid_from and (fact_row.get("meta") or {}).get("corrected"):
            text = (f"{subj} {present} {obj} — this was recorded in error "
                    f"and has been corrected")
        else:
            text = f"{subj} {past} {obj} ({_friendly_range(valid_from, valid_to)})"
    elif valid_from is not None and valid_from > now:
        text = f"From {friendly_date(valid_from)}, {subj} {future} {obj}"
    else:
        text = f"{subj} {present} {obj} (since {friendly_date(valid_from)})"

    if disputed:
        text += " (unconfirmed — conflicting information)"
    return _sentence(text)


def humanize_connection(fact_row: dict, entity_names: dict, *,
                        now: datetime | None = None) -> dict | None:
    """An entity↔entity fact -> {"other_id", "name", "description"} where the
    description reads "Connected to {name} — {plain sentence}". Returns None
    for value-object facts (no second entity to connect to)."""
    if fact_row.get("object_entity") is None:
        return None
    other_id = fact_row["object_entity"]
    other_name = _name_of(entity_names, other_id)
    sentence = humanize_fact(fact_row, entity_names, now=now)
    return {
        "other_id": str(other_id),
        "name": other_name,
        "description": f"Connected to {other_name} — {sentence}",
    }
