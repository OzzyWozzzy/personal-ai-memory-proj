"""Graphiti-native hybrid retrieval and source-grounded context assembly.

Graphiti is the only graph-memory authority.  This module never repairs,
supersedes, or mutates graph facts.  It composes a RAG response from:

* Graphiti's native BM25/vector/BFS candidate search;
* multi-query reciprocal-rank fusion;
* a local query-aware late-interaction reranker;
* exact source episodes attached to selected Graphiti facts; and
* short, query-centered raw passages from immutable Postgres episodes.

The last lane retrieves atomic memories first, then hydrates them with their
original ground truth. It
also provides recall when an extraction model omitted a detail.  Candidates are
scored before packing and raw passages are diversified by source, avoiding the
old behavior where one transcript prefix consumed the entire token budget.
"""
from __future__ import annotations

import asyncio
import math
import os
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import text as sql_text

from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.search.search_filters import (
    ComparisonOperator,
    DateFilter,
    SearchFilters,
)

from app.audit import write_audit
from app.engine import graph
from app.engine.native import aware, format_validity, value

_UTC = timezone.utc

DEFAULT_BUDGET = 1500
SEARCH_LIMIT = int(os.environ.get("MEMORY_SEARCH_LIMIT", "30"))
TOP_ENTITIES = int(os.environ.get("MEMORY_TOP_ENTITIES", "8"))
FINAL_EDGE_CAP = int(os.environ.get("MEMORY_FINAL_EDGE_CAP", "18"))
RAW_EPISODE_SCAN_LIMIT = int(os.environ.get("MEMORY_RAW_EPISODE_SCAN_LIMIT", "1000"))
RAW_RERANK_POOL = int(os.environ.get("MEMORY_RAW_RERANK_POOL", "80"))
RAW_FINAL_CAP = int(os.environ.get("MEMORY_RAW_FINAL_CAP", "10"))
RAW_PER_SOURCE = int(os.environ.get("MEMORY_RAW_PER_SOURCE", "5"))
RAW_LEXICAL_LANE_CAP = 8
STRONG_QUERY_COVERAGE = 0.60
ENTITY_SUMMARY_CAP = 3
COMMUNITY_SUMMARY_CAP = 2
MAX_QUERY_VARIANTS = 5


# ---------------------------------------------------------------------------
# Token accounting and hard-budget packer
# ---------------------------------------------------------------------------
_ENCODER = None


def _encoder():
    global _ENCODER
    if _ENCODER is None:
        import tiktoken

        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def estimate_tokens(s: str) -> int:
    return len(_encoder().encode(s)) if s else 0


def _line_cost(line: str, *, first: bool) -> int:
    return estimate_tokens(line if first else "\n" + line)


def _fit_passage(running, lines, prefix, content, token_budget):
    words = (content or "").split()
    if not words:
        return None
    first = not lines
    lo, hi, best = 1, len(words), None
    while lo <= hi:
        mid = (lo + hi) // 2
        line = prefix + " ".join(words[:mid])
        if running + _line_cost(line, first=first) <= token_budget:
            best = line
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _pack_items(items, token_budget):
    running = 0
    chosen, lines = [], []
    for item in items:
        first = not lines
        if item["kind"] == "passage":
            line = _fit_passage(
                running, lines, item["prefix"], item["content"], token_budget
            )
            if line is None:
                continue
            structured = dict(item["structured"])
            structured["snippet"] = line[len(item["prefix"]):]
            structured["line"] = line
        else:
            line = item["line"]
            if running + _line_cost(line, first=first) > token_budget:
                continue
            structured = item["structured"]
        running += _line_cost(line, first=first)
        lines.append(line)
        chosen.append(structured)

    block = "\n".join(lines)
    tokens = estimate_tokens(block)
    while tokens > token_budget and lines:
        lines.pop()
        chosen.pop()
        block = "\n".join(lines)
        tokens = estimate_tokens(block)
    return chosen, block, tokens


_run = graph._run_coro


# ---------------------------------------------------------------------------
# Query understanding: conservative mentions + inexpensive rewrites
# ---------------------------------------------------------------------------
_STOP = frozenset(
    "a an and are as at be by can could did do does for from had has have he her "
    "him his how i in is it its me my of on or she that the their them they this "
    "to was we were what when where which who whom whose why will with would you "
    "your about tell please all ever each many much total did does going know "
    "previous conversation conversations wanted want confirm information say said says "
    "mention mentioned".split()
    + " january february march april may june july august september october "
      "november december monday tuesday wednesday thursday friday saturday sunday "
      "valentine valentines christmas thanksgiving easter today yesterday tomorrow".split()
)
_CAP_RUN = re.compile(
    r"[A-Z][A-Za-z0-9]*(?:['\-]?[A-Za-z0-9]+)*"
    r"(?:\s+[A-Z][A-Za-z0-9]*(?:['\-]?[A-Za-z0-9]+)*)*"
)
_QUOTED = re.compile(r"['\"]([^'\"]{1,80})['\"]")
_TOKEN = re.compile(r"[a-z0-9]+")
_CLAUSE_SPLIT = re.compile(
    r"\s+(?:and then|as well as|versus|vs\.?|before|after|while|and)\s+|[?;,]",
    re.IGNORECASE,
)
_CURRENT_QUERY = re.compile(
    r"\b(current|currently|latest|now|today|present|most recent)\b",
    re.IGNORECASE,
)


def _sig_tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN.findall((text or "").casefold())
        if len(token) >= 3 and token not in _STOP
    }


def extract_mentions(query: str) -> list[str]:
    mentions: list[str] = []
    for match in _QUOTED.findall(query):
        if _sig_tokens(match):
            mentions.append(match)
    for run in _CAP_RUN.findall(query):
        words = run.split()
        while words and words[0].casefold() in _STOP:
            words.pop(0)
        phrase = " ".join(words)
        if _sig_tokens(phrase):
            mentions.append(phrase)
    return list(dict.fromkeys(mentions))


def _lexical_overlap(name: str, mention_tokens: set[str]) -> bool:
    return bool(_sig_tokens(name) & mention_tokens)


def rewrite_queries(query: str) -> list[str]:
    """Return bounded lexical/semantic variants for rank-fused search.

    This is intentionally LLM-free: the configured Graphiti LLM may be an
    offline fixture bridge, and retrieval must not create unrecorded write-time
    prompts.  The variants target the observed benchmark failure modes—named
    entities, multi-clause aggregation, and temporal anchors.
    """
    variants = [query.strip()]
    mentions = extract_mentions(query)
    variants.extend(mentions)

    clauses = [part.strip(" .?") for part in _CLAUSE_SPLIT.split(query)]
    clauses = [part for part in clauses if len(_sig_tokens(part)) >= 2]
    if len(clauses) > 1:
        variants.extend(clauses[:2])

    keywords = " ".join(
        token for token in _TOKEN.findall(query.casefold())
        if token not in _STOP and len(token) >= 3
    )
    if keywords and keywords.casefold() != query.strip().casefold():
        variants.append(keywords)

    if re.search(r"\b(when|before|after|between|how long|how many days|timeline)\b", query, re.I):
        variants.append(f"dates timeline {keywords}")
    if re.search(r"\b(total|how many|all|each|combined|altogether|evolve)\b", query, re.I):
        variants.append(f"all events values changes {keywords}")

    out = []
    seen = set()
    for variant in variants:
        normalized = " ".join(variant.split())
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            out.append(normalized)
        if len(out) >= MAX_QUERY_VARIANTS:
            break
    return out


# ---------------------------------------------------------------------------
# Native Graphiti search + rank fusion
# ---------------------------------------------------------------------------
def _date_groups(at: datetime, *, start: bool):
    if start:
        return [
            [DateFilter(date=at, comparison_operator=ComparisonOperator.less_than_equal)],
            [DateFilter(comparison_operator=ComparisonOperator.is_null)],
        ]
    return [
        [DateFilter(date=at, comparison_operator=ComparisonOperator.greater_than)],
        [DateFilter(comparison_operator=ComparisonOperator.is_null)],
    ]


def native_search_filters(*, as_of, as_believed_at, include_historical):
    """Build Graphiti ``SearchFilters``; no Python temporal repair/filter layer."""
    filters = SearchFilters()
    at = aware(as_of) or datetime.now(tz=_UTC)
    if include_historical:
        # Historical retrieval still excludes facts that had not started yet.
        filters.valid_at = _date_groups(at, start=True)
    else:
        filters.valid_at = _date_groups(at, start=True)
        filters.invalid_at = _date_groups(at, start=False)
    believed = aware(as_believed_at)
    if believed is not None:
        filters.created_at = _date_groups(believed, start=True)
        filters.expired_at = _date_groups(believed, start=False)
    return filters


async def _search_async(
    engine, query, config_name="COMBINED_HYBRID_SEARCH_CROSS_ENCODER",
    limit=SEARCH_LIMIT, *, filters=None, origins=None,
):
    recipe = graph.recipe(config_name)
    recipe.limit = int(limit)
    return await engine.graphiti.search_(
        query,
        recipe,
        group_ids=[engine.group_id],
        bfs_origin_node_uuids=origins,
        search_filter=filters or SearchFilters(),
    )


def _rrf_add(scores: dict[str, float], objects: dict[str, object], values, weight=1.0):
    for rank, obj in enumerate(values, start=1):
        uuid = str(obj.uuid)
        objects.setdefault(uuid, obj)
        scores[uuid] = scores.get(uuid, 0.0) + weight / (60.0 + rank)


async def _graph_candidates_async(engine, variants, filters):
    # Broad node/entity resolution first.  This preserves a clean native seed
    # set for Graphiti's BFS-capable advanced recipe.
    initial = await asyncio.gather(*[
        _search_async(
            engine, variant, "COMBINED_HYBRID_SEARCH_RRF", SEARCH_LIMIT,
            filters=filters,
        )
        for variant in variants
    ])

    node_scores: dict[str, float] = {}
    nodes_by_id: dict[str, object] = {}
    for result in initial:
        _rrf_add(node_scores, nodes_by_id, result.nodes)

    ranked_nodes = sorted(
        nodes_by_id.values(), key=lambda node: node_scores[str(node.uuid)], reverse=True
    )
    mention_tokens = set().union(*(_sig_tokens(m) for m in extract_mentions(variants[0]))) \
        if extract_mentions(variants[0]) else set()
    lexical_nodes = [
        node for node in ranked_nodes if _lexical_overlap(node.name, mention_tokens)
    ]
    resolved_nodes = (lexical_nodes or ranked_nodes)[:TOP_ENTITIES]
    origins = [str(node.uuid) for node in resolved_nodes] or None

    # One fully reranked native Graphiti expansion on the original question.
    # Rewrites already contribute broad RRF candidates above; rerunning the
    # cross encoder for every category of every rewrite multiplies latency
    # without adding a distinct candidate source.
    advanced = [await _search_async(
        engine,
        variants[0],
        "COMBINED_HYBRID_SEARCH_CROSS_ENCODER",
        SEARCH_LIMIT,
        filters=filters,
        origins=origins,
    )]

    edge_scores: dict[str, float] = {}
    edges_by_id: dict[str, object] = {}
    episode_scores: dict[str, float] = {}
    episodes_by_id: dict[str, object] = {}
    community_scores: dict[str, float] = {}
    communities_by_id: dict[str, object] = {}

    # Advanced, reranked candidates have higher fusion weight.  Initial RRF
    # candidates remain as a recall backstop.
    for result in initial:
        _rrf_add(edge_scores, edges_by_id, result.edges, weight=0.5)
        _rrf_add(episode_scores, episodes_by_id, result.episodes, weight=0.5)
    for result in advanced:
        _rrf_add(edge_scores, edges_by_id, result.edges)
        _rrf_add(node_scores, nodes_by_id, result.nodes)
        _rrf_add(episode_scores, episodes_by_id, result.episodes)
        _rrf_add(community_scores, communities_by_id, result.communities)

    def ranked(objects, scores):
        return sorted(objects.values(), key=lambda obj: scores[str(obj.uuid)], reverse=True)

    return {
        "edges": ranked(edges_by_id, edge_scores),
        "nodes": ranked(nodes_by_id, node_scores),
        "episodes": ranked(episodes_by_id, episode_scores),
        "communities": ranked(communities_by_id, community_scores),
        "resolved_nodes": resolved_nodes,
    }


async def _provenance_episodes_async(engine, edges):
    uuids = []
    seen = set()
    for edge in edges:
        for episode_uuid in value(edge, "episodes", []) or []:
            episode_uuid = str(episode_uuid)
            if episode_uuid not in seen:
                seen.add(episode_uuid)
                uuids.append(episode_uuid)
    if not uuids:
        return []
    try:
        return await EpisodicNode.get_by_uuids(engine.driver, uuids)
    except Exception:
        # One stale provenance id must not discard every valid source.  Graphiti's
        # bulk getter raises when none exist; individual reads recover partials.
        found = []
        for uuid in uuids:
            try:
                found.append(await EpisodicNode.get_by_uuid(engine.driver, uuid))
            except Exception:
                continue
        return found


# ---------------------------------------------------------------------------
# Raw source passages: speaker-aware/query-centered chunking + hybrid rerank
# ---------------------------------------------------------------------------
_TURN = re.compile(r"(?m)^(User|Assistant|System):\s*")


@dataclass
class RawPassage:
    passage_id: str
    episode_id: str
    source: str
    occurred_at: datetime | None
    role: str | None
    content: str
    score: float = 0.0
    query_coverage: float = 0.0
    lexical_score: float = 0.0
    provenance: bool = False


def _bounded_chunks(text: str, *, max_chars=900, overlap=120):
    text = " ".join((text or "").split())
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
            if boundary > start + max_chars // 2:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [chunk for chunk in chunks if chunk]


def _split_episode(row) -> list[RawPassage]:
    content = row.content or ""
    episode_id = str(row.episode_id)
    occurred_at = aware(row.occurred_at)
    source = row.source or ""
    meta = row.meta or {}
    parts: list[tuple[str | None, str]] = []
    matches = list(_TURN.finditer(content))
    if matches:
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
            parts.append((match.group(1).casefold(), content[match.end():end]))
    else:
        role = meta.get("role") if isinstance(meta, dict) else None
        parts.append((role, content))

    passages = []
    chunk_index = 0
    for role, part in parts:
        for chunk in _bounded_chunks(part):
            passages.append(RawPassage(
                passage_id=f"{episode_id}:{chunk_index}",
                episode_id=episode_id,
                source=source,
                occurred_at=occurred_at,
                role=role,
                content=chunk,
            ))
            chunk_index += 1
    return passages


def _raw_episode_rows(conn, tenant_id, *, as_of=None):
    where = "tenant_id = :tenant AND content IS NOT NULL"
    params = {"tenant": str(tenant_id), "limit": RAW_EPISODE_SCAN_LIMIT}
    if as_of is not None:
        where += " AND occurred_at <= :as_of"
        params["as_of"] = aware(as_of)
    return conn.execute(sql_text(
        "SELECT episode_id, source, occurred_at, content, meta FROM episodes "
        f"WHERE {where} ORDER BY occurred_at DESC, episode_id DESC LIMIT :limit"
    ), params).mappings().all()


def _cosine(left, right):
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denom) if denom else 0.0


def _lexical_score(query: str, passage: str) -> float:
    q = _sig_tokens(query)
    if not q:
        return 0.0
    p = _sig_tokens(passage)
    return len(q & p) / math.sqrt(max(len(q), 1) * max(len(p), 1))


def _term_forms(token: str) -> set[str]:
    """Return conservative forms used only by raw-passage lexical recall.

    Exact token sets remain authoritative for entity resolution and the
    false-premise guard.  Raw source retrieval needs a little more tolerance:
    questions routinely use a plural or a nearby derivation (``seat`` versus
    ``seats``, ``expansion`` versus ``expand``).  A five-character prefix is
    included only for terms at least six characters long, which catches those
    common variants without turning short, ambiguous words into matches.
    """
    forms = {token}
    if len(token) > 4 and token.endswith("ies"):
        forms.add(token[:-3] + "y")
    elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        forms.add(token[:-1])
    for form in tuple(forms):
        if len(form) >= 6:
            forms.add(f"prefix:{form[:5]}")
    return forms


def _query_coverage(query: str, passage: str) -> float:
    """Fraction of significant query terms present in a passage.

    Unlike cosine-style lexical overlap this does not penalize a long source
    turn.  It is especially useful for exact assistant-memory questions whose
    answer span repeats several rare words from the question.
    """
    query_terms = _sig_tokens(query)
    if not query_terms:
        return 0.0
    passage_forms = set().union(*(
        _term_forms(term) for term in _sig_tokens(passage)
    )) if passage else set()
    matched = sum(
        bool(_term_forms(term) & passage_forms) for term in query_terms
    )
    return matched / len(query_terms)


def _strong_query_match(query: str, coverage: float) -> bool:
    """Whether lexical evidence is strong enough to guard final recall.

    At least two significant terms (or the sole term in a one-word query) must
    match in addition to the coverage threshold.  Coverage ordering then keeps
    a repeated account-name-only passage below one that also matches the
    question's distinguishing terms.
    """
    term_count = len(_sig_tokens(query))
    if not term_count:
        return False
    matched = round(coverage * term_count)
    return coverage >= STRONG_QUERY_COVERAGE and matched >= min(2, term_count)


async def _rank_raw_passages_async(engine, variants, rows, provenance_ids):
    passages = [passage for row in rows for passage in _split_episode(row)]
    if not passages:
        return []
    for passage in passages:
        passage.provenance = passage.episode_id in provenance_ids

    query_vectors = await engine.graphiti.embedder.create_batch(variants)
    passage_vectors = await engine.graphiti.embedder.create_batch(
        [passage.content for passage in passages]
    )
    question = variants[0]
    wants_assistant = bool(re.search(r"\b(assistant|recommended|suggested|advised)\b", question, re.I))
    wants_current = bool(_CURRENT_QUERY.search(question))
    timestamps = [
        passage.occurred_at.timestamp()
        for passage in passages
        if passage.occurred_at is not None
    ]
    oldest = min(timestamps, default=0.0)
    recency_span = max(timestamps, default=0.0) - oldest
    for passage, vector in zip(passages, passage_vectors, strict=False):
        semantic = max(_cosine(query_vector, vector) for query_vector in query_vectors)
        lexical = max(_lexical_score(variant, passage.content) for variant in variants)
        coverage = _query_coverage(question, passage.content)
        passage.lexical_score = lexical
        passage.query_coverage = coverage
        # Most personal-memory questions target something the user said.  Make
        # speaker intent a meaningful signal so a long assistant explanation
        # containing generic query words (for example "video lectures") does
        # not outrank the user's actual event.  Assistant-targeted questions
        # invert the preference explicitly.
        role_bonus = 0.10 if passage.role == ("assistant" if wants_assistant else "user") else 0.0
        provenance_bonus = 0.08 if passage.provenance else 0.0
        recency_bonus = 0.0
        if wants_current and passage.occurred_at is not None:
            recency_bonus = (
                0.25 * (passage.occurred_at.timestamp() - oldest) / recency_span
                if recency_span else 0.25
            )
        passage.score = (
            semantic + 0.35 * lexical + 0.30 * coverage
            + role_bonus + provenance_bonus + recency_bonus
        )

    passages.sort(
        key=lambda passage: (passage.score, passage.passage_id), reverse=True
    )
    pool = passages[:RAW_RERANK_POOL]

    # Dense retrieval can produce many near-identical same-entity candidates.
    # Reserve a small lane for the strongest literal/inflectional matches so a
    # distinguishing answer span cannot be excluded before reranking.  The
    # overall model pool remains bounded at RAW_RERANK_POOL.
    lexical_lane = [
        passage
        for passage in sorted(
            passages,
            key=lambda passage: (
                passage.query_coverage,
                passage.lexical_score,
                passage.score,
                passage.passage_id,
            ),
            reverse=True,
        )
        if _strong_query_match(question, passage.query_coverage)
    ][:min(RAW_LEXICAL_LANE_CAP, RAW_RERANK_POOL)]
    pooled_ids = {passage.passage_id for passage in pool}
    replace_at = len(pool) - 1
    for passage in lexical_lane:
        if passage.passage_id in pooled_ids or replace_at < 0:
            continue
        pooled_ids.discard(pool[replace_at].passage_id)
        pool[replace_at] = passage
        pooled_ids.add(passage.passage_id)
        replace_at -= 1
    ranked_text = await engine.graphiti.cross_encoder.rank(
        question, [passage.content for passage in pool]
    )
    # Preserve duplicate text occurrences rather than collapsing provenance.
    by_text: dict[str, deque[RawPassage]] = defaultdict(deque)
    for passage in pool:
        by_text[passage.content].append(passage)
    reranked = []
    rerank_scores = [float(score) for _, score in ranked_text]
    low = min(rerank_scores, default=0.0)
    span = max(rerank_scores, default=0.0) - low
    for rank, (content, score) in enumerate(ranked_text, start=1):
        if by_text[content]:
            passage = by_text[content].popleft()
            # The previous implementation discarded the model score and added
            # only a ~0.05 rank nudge, leaving dense retrieval effectively in
            # control.  Normalize MaxSim within the candidate pool and give the
            # real reranker enough weight to change the final order.
            normalized = (float(score) - low) / span if span else 1.0 / rank
            passage.score += 0.45 * normalized
            reranked.append(passage)
    # The guard is intentionally binary and high precision.  Strong matches
    # order first by coverage and then by the semantic/reranker score; passages
    # outside the guard retain their normal hybrid score ordering.
    reranked.sort(
        key=lambda passage: (
            _strong_query_match(question, passage.query_coverage),
            (
                passage.query_coverage
                if _strong_query_match(question, passage.query_coverage)
                else 0.0
            ),
            passage.score,
            passage.lexical_score,
            passage.passage_id,
        ),
        reverse=True,
    )
    return reranked


def _diverse_passages(passages, *, prioritize_latest: bool = False):
    selected = []
    per_source: dict[str, int] = defaultdict(int)
    if prioritize_latest and passages:
        dated = [passage for passage in passages if passage.occurred_at is not None]
        if dated:
            newest = max(passage.occurred_at for passage in dated)
            # Put a small evidence bundle from the newest update before the
            # normal source-diversity pass. Otherwise diversity makes an old
            # contradictory episode the second line of every current-state
            # answer, even when its rank is much lower.
            for passage in passages:
                if passage.occurred_at != newest:
                    continue
                selected.append(passage)
                per_source[passage.episode_id] += 1
                if per_source[passage.episode_id] >= min(3, RAW_PER_SOURCE):
                    break
    # First pass maximizes distinct support episodes; second pass permits one
    # or more additional passages per source for a local multi-turn answer.
    for allowance in range(1, RAW_PER_SOURCE + 1):
        for passage in passages:
            if len(selected) >= RAW_FINAL_CAP:
                return selected
            if passage in selected or per_source[passage.episode_id] >= allowance:
                continue
            selected.append(passage)
            per_source[passage.episode_id] += 1
    return selected


# ---------------------------------------------------------------------------
# Rendering and response shaping
# ---------------------------------------------------------------------------
def _short(uuid) -> str:
    return str(uuid)[:8]


def _edge_item(edge):
    window = format_validity(edge) or "timeless"
    episode_ids = [str(episode_id) for episode_id in (value(edge, "episodes", []) or [])]
    provenance_count = len(episode_ids)
    line = f"fact:{_short(edge.uuid)} | {edge.fact} | {window} | sources:{provenance_count}"
    return {
        "kind": "fixed",
        "line": line,
        "structured": {
            "ref": _short(edge.uuid),
            "uuid": str(edge.uuid),
            "fact": edge.fact,
            "episode_ids": episode_ids,
            "window": window,
            "stratum": "temporal" if format_validity(edge) else "semantic",
            "older_value": False,
            "line": line,
        },
    }


def _entity_item(node):
    line = f"entity:{node.name} | {node.summary}"
    return {
        "kind": "fixed",
        "line": line,
        "structured": {
            "ref": f"entity:{node.uuid}",
            "entity_id": str(node.uuid),
            "name": node.name,
            "summary": node.summary,
            "stratum": "semantic",
            "line": line,
        },
    }


def _community_item(community):
    line = f"community:{community.name} | {community.summary}"
    return {
        "kind": "fixed",
        "line": line,
        "structured": {
            "ref": f"community:{community.uuid}",
            "community_id": str(community.uuid),
            "name": community.name,
            "summary": community.summary,
            "stratum": "semantic",
            "line": line,
        },
    }


def _passage_item(passage: RawPassage):
    date = passage.occurred_at.date().isoformat() if passage.occurred_at else ""
    role = passage.role or passage.source or "source"
    prefix = f"source:{_short(passage.episode_id)} | {role} | {date} | "
    return {
        "kind": "passage",
        "prefix": prefix,
        "content": passage.content,
        "structured": {
            "ref": f"source:{passage.passage_id}",
            "episode_id": passage.episode_id,
            "passage_id": passage.passage_id,
            "source": passage.source,
            "role": passage.role,
            "occurred_at": passage.occurred_at.isoformat() if passage.occurred_at else None,
            "provenance": passage.provenance,
            "stratum": "semantic",
        },
    }


def _graph_episode_item(episode):
    date = aware(episode.valid_at)
    prefix = (
        f"source:{_short(episode.uuid)} | {episode.source_description or 'episode'} | "
        f"{date.date().isoformat() if date else ''} | "
    )
    return {
        "kind": "passage",
        "prefix": prefix,
        "content": episode.content or "",
        "structured": {
            "ref": f"source:{episode.uuid}",
            "episode_id": str(episode.uuid),
            "source": episode.source_description or "",
            "occurred_at": date.isoformat() if date else None,
            "provenance": True,
            "stratum": "semantic",
        },
    }


def _edge_in_scope(edge, scopes):
    return (
        str(value(edge, "source_node_uuid")) in scopes
        or str(value(edge, "target_node_uuid")) in scopes
    )


def _response(route, items, block, tokens, token_budget, fmt):
    mix = {"temporal": 0, "semantic": 0}
    for item in items:
        if item.get("stratum") in mix:
            mix[item["stratum"]] += 1
    response = {
        "route": route,
        "items_returned": len(items),
        "tokens_estimated": tokens,
        "budget": token_budget,
        "stratum_mix": mix,
        "cache_stable_hash": None,
    }
    response["items" if fmt == "json" else "block"] = items if fmt == "json" else block
    return response


def _audit(conn, tenant_id, actor, response, query, route, overridden):
    write_audit(conn, tenant_id, actor, "retrieval.route", {
        "route": route,
        "overridden": overridden,
        "query": query,
        "tokens": response["tokens_estimated"],
        "budget": response["budget"],
        "stratum_mix": response["stratum_mix"],
        "cache_stable_hash": None,
    })


def assemble_context(
    conn,
    tenant_id,
    *,
    query,
    subject_scope=None,
    token_budget=DEFAULT_BUDGET,
    as_of=None,
    as_believed_at=None,
    include_disputed=False,
    include=None,
    fmt="block",
    actor="retrieval",
    allowed_subject_scopes=None,
    interval_op=None,
    ref_interval=None,
    include_historical=False,
):
    """Assemble a scored, source-grounded context packet under a hard budget.

    ``include_disputed`` remains accepted for compatibility; Graphiti has no
    separate dispute state.  Allen ``interval_op``/``ref_interval`` are rejected
    by the API in Graphiti-native mode and ignored here for direct legacy callers.
    """
    del include_disputed, interval_op, ref_interval
    query = (query or "").strip()
    overridden = include in ("facts", "knowledge", "narrative", "merged")
    if not query or token_budget <= 0:
        response = _response("facts", [], "", 0, token_budget, fmt)
        _audit(conn, tenant_id, actor, response, query, "facts", overridden)
        return response

    # Self-configure the process-wide clients on first use (mirrors
    # views._engine / ingest._engine) so a fresh process whose FIRST request
    # is /v1/context works instead of raising "no clients configured".
    try:
        engine = graph.get_engine(str(tenant_id))
    except RuntimeError:
        from app.engine.ingest import _configure_default_clients

        _configure_default_clients()
        engine = graph.get_engine(str(tenant_id))
    variants = rewrite_queries(query)
    filters = native_search_filters(
        as_of=as_of,
        as_believed_at=as_believed_at,
        include_historical=include_historical,
    )
    candidates = _run(_graph_candidates_async(engine, variants, filters))

    scopes = None if allowed_subject_scopes is None else {
        str(scope) for scope in allowed_subject_scopes
    }
    edges = list(candidates["edges"])
    if scopes is not None:
        edges = [edge for edge in edges if _edge_in_scope(edge, scopes)]
    edges = edges[:FINAL_EDGE_CAP]

    # A caller-provided subject is an explicit BFS anchor even when its name did
    # not rank for the query.  Run one native expansion and merge it at the tail.
    if subject_scope is not None and str(subject_scope) not in {
        str(node.uuid) for node in candidates["resolved_nodes"]
    }:
        expanded = _run(_search_async(
            engine,
            query,
            limit=SEARCH_LIMIT,
            filters=filters,
            origins=[str(subject_scope)],
        ))
        seen_edges = {str(edge.uuid) for edge in edges}
        for edge in expanded.edges:
            if str(edge.uuid) not in seen_edges and (
                scopes is None or _edge_in_scope(edge, scopes)
            ):
                edges.append(edge)
                seen_edges.add(str(edge.uuid))
            if len(edges) >= FINAL_EDGE_CAP:
                break

    provenance_episodes = _run(_provenance_episodes_async(engine, edges))
    provenance_ids = {
        str(episode.uuid) for episode in provenance_episodes
    } | {
        str(episode.name) for episode in provenance_episodes
    }

    # False-premise guard after broad retrieval: an explicit named entity must
    # resolve either to a graph node or literally in source evidence.  This keeps
    # broad semantic retrieval from fabricating support for a nonexistent name.
    mentions = extract_mentions(query)
    mention_tokens = set().union(*(_sig_tokens(m) for m in mentions)) if mentions else set()
    resolved_name = any(
        _lexical_overlap(node.name, mention_tokens) for node in candidates["nodes"]
    ) if mentions else True

    raw_rows = [] if scopes is not None else _raw_episode_rows(
        conn, tenant_id, as_of=as_of
    )
    raw_literal_match = any(
        mention_tokens & _sig_tokens(row.content or "") for row in raw_rows
    ) if mentions else True
    if mentions and not raw_literal_match:
        raw_literal_match = any(
            mention_tokens & _sig_tokens(getattr(episode, "content", "") or "")
            for episode in candidates["episodes"]
        )
    if mentions and not resolved_name and not raw_literal_match:
        response = _response("facts", [], "", 0, token_budget, fmt)
        _audit(conn, tenant_id, actor, response, query, "facts", overridden)
        return response

    raw_passages = []
    if raw_rows:
        raw_passages = _run(_rank_raw_passages_async(
            engine, variants, raw_rows, provenance_ids
        ))
        raw_passages = _diverse_passages(
            raw_passages,
            prioritize_latest=bool(_CURRENT_QUERY.search(query)),
        )

    only_sources = include == "knowledge"
    want_facts = include not in ("knowledge",)
    want_sources = include in (None, "knowledge", "merged")

    fact_items = []
    if want_facts:
        fact_items.extend(_edge_item(edge) for edge in edges)
        summaries = [
            node for node in candidates["nodes"]
            if (getattr(node, "summary", "") or "").strip()
        ][:ENTITY_SUMMARY_CAP]
        fact_items.extend(_entity_item(node) for node in summaries)
        communities = [
            community for community in candidates["communities"]
            if (getattr(community, "summary", "") or "").strip()
        ][:COMMUNITY_SUMMARY_CAP]
        fact_items.extend(_community_item(community) for community in communities)

    source_items = []
    if want_sources:
        source_items.extend(_passage_item(passage) for passage in raw_passages)
        # If Postgres raw content is unavailable (governed deletion/import), keep
        # Graphiti provenance usable rather than returning no source evidence.
        raw_ids = {passage.episode_id for passage in raw_passages}
        for episode in provenance_episodes + list(candidates["episodes"]):
            if str(episode.uuid) in raw_ids or str(episode.name) in raw_ids:
                continue
            source_items.append(_graph_episode_item(episode))
            if len(source_items) >= RAW_FINAL_CAP:
                break

    # Interleave facts and sources so neither channel can monopolize the budget.
    items = []
    if only_sources:
        items = source_items
    else:
        for index in range(max(len(fact_items), len(source_items))):
            if index < len(fact_items):
                items.append(fact_items[index])
            if index < len(source_items):
                items.append(source_items[index])

    chosen, block, tokens = _pack_items(items, token_budget)
    has_fact = any("uuid" in item or "entity_id" in item for item in chosen)
    has_source = any("episode_id" in item for item in chosen)
    if overridden:
        route = "facts" if include == "narrative" else include
    elif has_fact and has_source:
        route = "merged"
    elif has_source:
        route = "knowledge"
    else:
        route = "facts"

    response = _response(route, chosen, block, tokens, token_budget, fmt)
    _audit(conn, tenant_id, actor, response, query, route, overridden)
    return response
