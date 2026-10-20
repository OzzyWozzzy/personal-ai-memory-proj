"""Append-only audit log writer (P6: everything is auditable)."""
from __future__ import annotations

import json

from sqlalchemy import text


def write_audit(conn, tenant_id, actor: str, action: str, detail: dict) -> None:
    """INSERT one audit_log row. The table is append-only by grants:
    app_user has INSERT and SELECT only — no UPDATE, no DELETE."""
    conn.execute(
        text(
            """
            INSERT INTO audit_log (tenant_id, actor, action, detail)
            VALUES (:tenant_id, :actor, :action, CAST(:detail AS jsonb))
            """
        ),
        {
            "tenant_id": str(tenant_id),
            "actor": actor,
            "action": action,
            "detail": json.dumps(detail),
        },
    )
