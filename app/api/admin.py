"""Token-efficiency metrics for the governance dashboard.

``GET /v1/admin/token-metrics`` (spec name) and its back-compat alias
``GET /v1/admin/token_stats`` aggregate the ``retrieval.route`` audit rows for
the tenant into a workspace token distribution: query count, mean, p50/p95,
and a per-route breakdown.
"""
from __future__ import annotations

import math

from fastapi import APIRouter, Request
from sqlalchemy import text

from app import db

router = APIRouter(prefix="/v1/admin", tags=["admin"])


def _percentile(sorted_vals: list[int], pct: float) -> int:
    """Nearest-rank percentile (deterministic, no interpolation). Empty -> 0."""
    if not sorted_vals:
        return 0
    k = max(0, math.ceil(pct / 100 * len(sorted_vals)) - 1)
    return sorted_vals[k]


def _token_metrics(tenant_id) -> dict:
    with db.tenant_connection(tenant_id) as conn:
        rows = conn.execute(
            text(
                "SELECT detail FROM audit_log "
                "WHERE tenant_id = :t AND action = 'retrieval.route'"
            ),
            {"t": str(tenant_id)},
        ).mappings().all()

    by_route = {"facts": 0, "knowledge": 0, "narrative": 0, "merged": 0}
    stratum_mix = {"temporal": 0, "semantic": 0}
    tokens: list[int] = []
    for r in rows:
        detail = r["detail"]
        route = detail.get("route")
        if route in by_route:
            by_route[route] += 1
        # ``tokens`` is the v3.2 field; fall back to the legacy name for old rows.
        tok = detail.get("tokens", detail.get("tokens_estimated"))
        if tok is not None:
            tokens.append(int(tok))
        mix = detail.get("stratum_mix") or {}
        stratum_mix["temporal"] += int(mix.get("temporal", 0) or 0)
        stratum_mix["semantic"] += int(mix.get("semantic", 0) or 0)

    tokens.sort()
    avg = round(sum(tokens) / len(tokens), 2) if tokens else 0.0
    return {
        "queries": len(rows),
        "avg_tokens": avg,
        "p50_tokens": _percentile(tokens, 50),
        "p95_tokens": _percentile(tokens, 95),
        "by_route": by_route,
        "stratum_mix": stratum_mix,
    }


@router.get("/token-metrics")
def token_metrics(request: Request) -> dict:
    """Spec-named token-efficiency metrics."""
    return _token_metrics(request.state.tenant_ctx.tenant_id)


@router.get("/token_stats")
def token_stats(request: Request) -> dict:
    """Back-compat alias of ``/v1/admin/token-metrics`` (same payload)."""
    return _token_metrics(request.state.tenant_ctx.tenant_id)
