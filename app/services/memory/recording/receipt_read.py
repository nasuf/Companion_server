"""Content-free administration of durable extraction receipts."""

from datetime import datetime, timezone
from app.db import db
from app.services.memory.recording.receipts import FAILURE_REASONS


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
        # Preserve the original three-value contract. Older Web releases treat
        # every non-processing error as source change/deletion.
        row["last_error"] = (
            row["last_error"]
            if row["last_error"] in {"source_changed", "source_deleted"}
            else "processing_failed" if row["last_error"] else None
        )
    return {
        "records": rows,
        "limit": limit,
        "sampled_at": datetime.now(timezone.utc).isoformat(),
        "recovery": "next_chat_trigger",
        "scope": "active_within_recent_batches",
        "denominator": "this_page_only",
    }


def message_outcome(row):
    """Completion is proven by durable source/version evidence, never a cursor."""
    if row.get("batch_id") is None:
        return "unrecorded"
    if row.get("source_matches") is not True or row.get("batch_status") == "blocked":
        return "needs_reconcile"
    if row.get("batch_status") in {"preparing", "processing"}:
        return "retryable_failed" if row.get("last_error") else "processing"
    if row.get("batch_status") == "completed" and row.get("processing_outcome") in {
        "processed", "no_fact", "policy_skipped"
    }:
        return row["processing_outcome"]
    return "historical_unknown"


# Operational visibility only: strict ordered recovery is unchanged (M02.05).
STALE_FAILURE_HOURS = 2
STALE_FAILURE_LIMIT = 5


def _utc_iso(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value.replace(tzinfo=value.tzinfo or timezone.utc).astimezone(timezone.utc).isoformat()


async def stalled_batch_monitor():
    """Count active unfinished batches, independent of the recent-message page.

    At most five IDs/reason codes leave SQL. A 3s statement timeout bounds the
    aggregate scan; failure is explicit, never reported as a zero backlog.
    """
    sampled_at = datetime.now(timezone.utc).isoformat()
    try:
        async with db.tx() as tx:
            await tx.execute_raw("SET LOCAL statement_timeout = '3s'")
            rows = await tx.query_raw('''WITH active AS MATERIALIZED (
                SELECT b.id,b.updated_at,b.last_error FROM memory_extraction_batches b
                JOIN conversations c ON c.id=b.conversation_id AND c.user_id=b.user_id
                    AND c.workspace_id=b.workspace_id
                JOIN chat_workspaces w ON w.id=b.workspace_id AND w.user_id=b.user_id AND w.agent_id=c.agent_id
                JOIN ai_agents a ON a.id=c.agent_id AND a.user_id=c.user_id
                WHERE b.status IN ('preparing','processing') AND NOT c.is_deleted
                    AND c.archived_at IS NULL AND w.status='active'
            ), stale AS MATERIALIZED (
                SELECT * FROM active WHERE last_error IS NOT NULL
                    AND updated_at < CURRENT_TIMESTAMP-($1::int*INTERVAL '1 hour')
            ) SELECT (SELECT count(*)::int FROM active) AS unfinished_batches,
                (SELECT count(*)::int FROM active WHERE last_error IS NOT NULL) AS failed_batches,
                (SELECT count(*)::int FROM stale) AS stale_failed_batches,
                s.id,s.updated_at,s.last_error
            FROM (SELECT 1) seed LEFT JOIN
                (SELECT * FROM stale ORDER BY updated_at,id LIMIT $2) s ON true
            ORDER BY s.updated_at,s.id''', STALE_FAILURE_HOURS, STALE_FAILURE_LIMIT)
        totals = rows[0]
        return {'status': 'measured', 'sampled_at': sampled_at,
                'threshold_hours': STALE_FAILURE_HOURS, 'limit': STALE_FAILURE_LIMIT,
                'scope': 'all_active_conversation_sides', 'window': 'all_retained_unfinished_batches',
                'denominator': 'unfinished_batches',
                **{key: totals[key] for key in ('unfinished_batches','failed_batches','stale_failed_batches')},
                'oldest': [{'batch_id': row['id'], 'updated_at': _utc_iso(row['updated_at']),
                            'reason': row['last_error'] if row['last_error'] in FAILURE_REASONS else 'processing_failed'}
                           for row in rows if row['id'] is not None]}
    except Exception:
        # SQL/connection failures contain operational secrets; never echo them.
        return {'status': 'unavailable', 'sampled_at': sampled_at,
                'threshold_hours': STALE_FAILURE_HOURS, 'limit': STALE_FAILURE_LIMIT}


async def recent_message_processing(*, limit=20, hours=24):
    """Bounded recent-message coverage sample; not a fleet-wide success rate."""
    if type(limit) is not int or not 1 <= limit <= 50 or type(hours) is not int or not 1 <= hours <= 168:
        raise ValueError("invalid_message_coverage_page")
    rows = await db.query_raw('''WITH sample AS (
        SELECT id,conversation_id,role,content,metadata,created_at FROM messages
        WHERE created_at>=(CURRENT_TIMESTAMP-($2::int*INTERVAL '1 hour')) AT TIME ZONE 'UTC'
          AND role IN ('user','assistant')
        ORDER BY created_at DESC,id DESC LIMIT $1
    ) SELECT m.id AS message_id,m.conversation_id,(m.created_at AT TIME ZONE 'UTC') AS message_at,
        m.role AS _role,m.content AS _content,m.metadata AS _metadata,
        CASE m.role WHEN 'user' THEN 'user' ELSE 'ai' END AS side,
        b.id AS batch_id,b.status AS batch_status,b.pipeline_version,b.processing_outcome,
        b.outcome_reason,b.last_error,b.updated_at,b.fact_count AS batch_fact_count,
        jsonb_array_length(b.sources) AS batch_source_count,b.extraction_provenance,
        EXISTS(SELECT 1 FROM jsonb_array_elements(b.sources) x
            WHERE x->>'ref'=m.id AND x->>'expected_version'=encode(sha256(convert_to(m.content,'UTF8')),'hex')
              AND (x->>'expected_role')=m.role
              AND (x->>'expected_conversation_id')=m.conversation_id
              AND (x->>'expected_statement_time')::timestamptz=(m.created_at AT TIME ZONE 'UTC')
              AND (x->>'expected_card_version' IS NULL OR x->>'expected_card_version'=
                   encode(sha256(convert_to((m.metadata->'component_card')::text,'UTF8')),'hex'))
        ) AS source_matches
        FROM sample m JOIN conversations c ON c.id=m.conversation_id
        JOIN chat_workspaces w ON w.id=c.workspace_id AND w.user_id=c.user_id AND w.agent_id=c.agent_id
        JOIN ai_agents a ON a.id=c.agent_id AND a.user_id=c.user_id
        LEFT JOIN memory_extraction_sources s ON s.message_id=m.id AND s.side=CASE m.role WHEN 'user' THEN 'user' ELSE 'ai' END
        LEFT JOIN memory_extraction_batches b ON b.id=s.batch_id AND b.user_id=c.user_id
            AND b.workspace_id=c.workspace_id AND b.conversation_id=c.id
        WHERE NOT c.is_deleted AND c.archived_at IS NULL AND w.status='active'
        ORDER BY m.created_at DESC,m.id DESC''', limit, hours)
    counts = {}
    for row in rows:
        # Reuse the write-side eligibility rules, including gift/status cards.
        # These private fields are removed before any response is constructed.
        source = {name: row.pop('_' + name) for name in ('role', 'content', 'metadata')}
        from app.services.memory.recording.chat_sources import excluded_from_chat_extraction, extraction_text
        try:
            reserved = excluded_from_chat_extraction(source) or not extraction_text(source).strip()
            row['source_matches'] = bool(row['source_matches']) and (
                reserved == (row['outcome_reason'] == 'reserved_messages'))
        except (RuntimeError, ValueError, TypeError):
            row['source_matches'] = False
        row['outcome'] = message_outcome(row)
        row['message_at'] = _utc_iso(row['message_at'])
        row['updated_at'] = _utc_iso(row['updated_at'])
        counts[row['outcome']] = counts.get(row['outcome'], 0) + 1
        row['last_error'] = row['last_error'] if row['last_error'] in FAILURE_REASONS else None
        allowed = {'processed','empty_extraction','heuristic_filtered','prefilter_rejected',
                   'facts_not_written','reserved_messages'}
        row['outcome_reason'] = row['outcome_reason'] if row['outcome_reason'] in allowed else None
        raw = row.pop('extraction_provenance')
        row['provenance'] = {k: raw.get(k) for k in ('stage','schema_version','requested_model',
            'actual_model','template_key','template_sha256','source_contract_sha256')} if isinstance(raw, dict) else None
    return {'records': rows, 'counts': counts, 'sample_size': len(rows), 'limit': limit,
            'window_hours': hours, 'sampled_at': datetime.now(timezone.utc).isoformat(),
            'denominator': 'this_page_only', 'scope': 'active_within_recent_messages',
            'recovery': 'next_chat_trigger', 'replay_supported': False,
            'stalled_batches': await stalled_batch_monitor()}
