"""GET /v1/stats — compact counts for the local workspace dashboard."""
from __future__ import annotations

from fastapi import APIRouter, Query, Request
from sqlalchemy import text

from app.db import tenant_connection
from app.engine import views

router = APIRouter()

# episodes/entities stay in Postgres; facts come from the
# graph (RELATES_TO edges); Postgres remains the immutable source ledger.
_COUNTS = text("""
    SELECT
      (SELECT count(*) FROM episodes) AS episodes,
      (SELECT count(*) FROM entities) AS entities
""")

# Ingestion activity: episodes recorded per day, zero-filled so the chart has
# one bar per day even on quiet days. recorded_at (transaction time) is the
# honest ingestion axis — occurred_at is the *content's* time.
_INGEST_BY_DAY = text("""
    SELECT d::date AS day, count(e.episode_id) AS episodes
      FROM generate_series(current_date - (:days - 1) * interval '1 day',
                           current_date, interval '1 day') AS d
      LEFT JOIN episodes e ON e.recorded_at::date = d::date
     GROUP BY 1 ORDER BY 1
""")


@router.get("/v1/stats")
def stats(request: Request,
          days: int = Query(0, ge=0, le=365)) -> dict:
    """Counts, optionally with a per-day ingestion series."""
    ctx = request.state.tenant_ctx
    # facts/facts_current from the graph (all edges / open edges).
    facts, facts_current = views.fact_counts(ctx.tenant_id)
    with tenant_connection(ctx.tenant_id) as conn:
        row = conn.execute(_COUNTS).one()
        series = None
        if days:
            series = [{"day": r.day.isoformat(), "episodes": r.episodes}
                      for r in conn.execute(_INGEST_BY_DAY, {"days": days})]
    out = {"episodes": row.episodes, "entities": row.entities,
           "facts": facts, "facts_current": facts_current}
    if series is not None:
        out["ingest_by_day"] = series
    return out
