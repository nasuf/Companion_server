"""Content-free administration of durable extraction receipts."""

from datetime import datetime, timezone
from app.db import db


async def recent_receipts(*, limit=20):
    if not 1 <= limit <= 50:
        raise ValueError("invalid_receipt_page")
    rows = await db.query_raw(
        """SELECT b.id,b.conversation_id,b.side,b.status,b.prepared_at,b.updated_at,
        b.last_error,b.fact_count,jsonb_array_length(b.sources) AS source_count,
        r.written,r.skipped,r.erased,r.followups_pending,r.followups_superseded
        FROM (SELECT * FROM memory_extraction_batches ORDER BY prepared_at DESC,id DESC LIMIT $1) b
        JOIN conversations c ON c.id=b.conversation_id AND c.user_id=b.user_id AND c.workspace_id=b.workspace_id
        JOIN chat_workspaces w ON w.id=b.workspace_id AND w.user_id=b.user_id AND w.agent_id=c.agent_id
        LEFT JOIN LATERAL (SELECT count(*) FILTER(WHERE status='written')::int AS written,
            count(*) FILTER(WHERE status='skipped')::int AS skipped,
            count(*) FILTER(WHERE status='erased')::int AS erased,
            count(*) FILTER(WHERE followup_outcome='superseded')::int AS followups_superseded,
            count(*) FILTER(WHERE status='written' AND NOT followup_done)::int AS followups_pending
            FROM memory_write_receipts WHERE batch_id=b.id) r ON true
        WHERE NOT c.is_deleted AND c.archived_at IS NULL AND w.status='active'
        ORDER BY b.prepared_at DESC,b.id DESC""",
        limit,
    )
    for row in rows:
        row["last_error"] = (
            row["last_error"]
            if row["last_error"]
            in {"processing_failed", "source_changed", "source_deleted"}
            else None
        )
    return {
        "records": rows,
        "limit": limit,
        "sampled_at": datetime.now(timezone.utc).isoformat(),
        "recovery": "next_chat_trigger",
        "scope": "active_within_recent_batches",
        "denominator": "this_page_only",
    }
