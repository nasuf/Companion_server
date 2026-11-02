"""Atomic replacement metadata and source-revalidated, bounded history reads."""

from datetime import datetime, timezone
import json

from app.services.memory.manual_write import MemoryWriteError


async def current_version_id(database, record) -> int:
    """Latest semantic revision, writing the labelled baseline a legacy row lacks.

    Memories created before temporal versions existed have no row until their
    first semantic update (the migration does no backfill). Insert the same
    'baseline' the capture trigger would write, inside the caller's transaction
    and under its row lock, rather than failing every legacy confirmation.
    """
    rows = await database.query_raw(
        """SELECT id FROM memory_temporal_versions WHERE memory_id=$1 AND side='user'
        AND user_id=$2 AND workspace_id=$3 ORDER BY id DESC LIMIT 1""",
        record.id, record.userId, record.workspaceId,
    )
    if rows:
        return rows[0]["id"]
    rows = await database.query_raw(
        """INSERT INTO memory_temporal_versions(memory_id,side,user_memory_id,ai_memory_id,
          user_id,workspace_id,content,content_version,occur_time,statement_time,is_archived,record_kind)
        SELECT m.id,'user',m.id,NULL,m.user_id,m.workspace_id,m.content,
          encode(sha256(convert_to(m.content,'UTF8')),'hex'),
          m.occur_time AT TIME ZONE 'UTC',m.statement_time AT TIME ZONE 'UTC',m.is_archived,'baseline'
        FROM memories_user m WHERE m.id=$1 AND m.user_id=$2 AND m.workspace_id=$3
          AND NOT EXISTS (SELECT 1 FROM memory_temporal_versions v WHERE v.side='user' AND v.memory_id=m.id)
        RETURNING id""",
        record.id, record.userId, record.workspaceId,
    )
    if not rows:
        raise MemoryWriteError(409, "memory_change_revision_unavailable")
    return rows[0]["id"]


async def persist_change(database, *, old, new, origin_id, old_version_id, semantics):
    revisions = await database.query_raw(
        """SELECT id FROM memory_temporal_versions WHERE memory_id=$1 AND side='user'
        AND user_id=$2 AND workspace_id=$3 ORDER BY id DESC LIMIT 1""",
        new.id,
        new.userId,
        new.workspaceId,
    )
    if not old_version_id or not revisions or semantics.statement_at is None:
        raise MemoryWriteError(409, "memory_change_revision_unavailable")
    if semantics.kind in {"transition", "correction"}:
        # The current projection and the existing chain commit in the same
        # transaction. Do not await a postcommit best-effort quality refresh.
        rows = await database.query_raw(
            """INSERT INTO memory_quality_states
            (memory_id,memory_source,user_id,workspace_id,superseded_by_memory_id,signals)
            VALUES ($1,'user',$2,$3,$4,'{}'::jsonb)
            ON CONFLICT(memory_id,memory_source) DO UPDATE
              SET superseded_by_memory_id=EXCLUDED.superseded_by_memory_id,updated_at=CURRENT_TIMESTAMP
            WHERE memory_quality_states.user_id=EXCLUDED.user_id
              AND memory_quality_states.workspace_id=EXCLUDED.workspace_id
              AND (memory_quality_states.superseded_by_memory_id IS NULL
                   OR memory_quality_states.superseded_by_memory_id=EXCLUDED.superseded_by_memory_id)
            RETURNING memory_id""",
            old.id,
            old.userId,
            old.workspaceId,
            new.id,
        )
        if not rows:
            raise MemoryWriteError(409, "memory_change_replacement_conflict")
    value = semantics.diagnostic()
    await database.execute_raw(
        """INSERT INTO memory_fact_changes
        (id,user_id,workspace_id,old_memory_id,new_memory_id,old_version_id,new_version_id,
         kind,semantics,statement_at,boundary_start,boundary_end,boundary_source_id)
        VALUES ($1,$2,$3,$4,$5,$6::bigint,$7::bigint,$8,$9::jsonb,
                $10::timestamptz,$11::timestamptz,$12::timestamptz,$13)""",
        origin_id,
        old.userId,
        old.workspaceId,
        old.id,
        new.id,
        old_version_id,
        revisions[0]["id"],
        semantics.kind,
        json.dumps(value),
        value["statement_at"],
        value["boundary"]["start"],
        value["boundary"]["end_exclusive"],
        semantics.boundary_source_id,
    )

    if semantics.kind == "correction":
        await database.execute_raw(
            """INSERT INTO memory_source_retractions
            (source_message_id,source_version,source_statement_time,user_id,workspace_id,memory_id,origin_id)
            SELECT DISTINCT m.id,e.source_version,e.source_statement_time,e.user_id,e.workspace_id,e.memory_id,$1
            FROM memory_evidence_links e JOIN messages m ON m.id=e.source_message_id
            JOIN conversations c ON c.id=m.conversation_id
            WHERE e.memory_id=$2 AND e.memory_source='user' AND e.user_id=$3 AND e.workspace_id=$4
              AND e.relation IN ('extracted_from','recorded_from') AND m.role='user'
              AND COALESCE(e.temporal_basis->>'source_role','fact')='fact'
              AND c.user_id=$3 AND c.workspace_id=$4
              AND e.source_version=encode(sha256(convert_to(m.content,'UTF8')),'hex')
              AND e.source_statement_time=m.created_at
            ON CONFLICT(source_message_id,source_version,memory_id) DO NOTHING""",
            origin_id,
            old.id,
            old.userId,
            old.workspaceId,
        )


async def _read_change_rows(
    database, *, user_id, workspace_id, memory_ids, limit, known_at=None
):
    if not 1 <= len(memory_ids) <= 160 or not 1 <= limit <= 320:
        raise ValueError("invalid_memory_change_page")
    if known_at is not None and (
        not isinstance(known_at, datetime) or known_at.tzinfo is None
    ):
        raise ValueError("memory_change_requires_timezone")
    rows = await database.query_raw(
        """SELECT f.id,f.old_memory_id,f.new_memory_id,f.kind,f.semantics,f.recorded_at,
          f.old_version_id,f.new_version_id,
          (o.statement_time AT TIME ZONE 'UTC') AS old_statement_at,
          (jsonb_array_length(i.payload_text::jsonb->'input'->'sources') BETWEEN 1 AND 80
           AND NOT EXISTS (
            SELECT 1 FROM jsonb_array_elements(i.payload_text::jsonb->'input'->'sources') s
            LEFT JOIN messages m ON m.id=s->>'ref'
            LEFT JOIN conversations c ON c.id=m.conversation_id
            WHERE s->>'kind'<>'message' OR m.id IS NULL OR m.role<>'user'
              OR c.user_id<>f.user_id OR c.workspace_id<>f.workspace_id OR c.agent_id<>w.agent_id
              OR c.is_deleted OR c.archived_at IS NOT NULL
              OR s->>'expected_version' IS DISTINCT FROM encode(sha256(convert_to(m.content,'UTF8')),'hex')
              OR (s->>'expected_statement_time')::timestamptz IS DISTINCT FROM m.created_at AT TIME ZONE 'UTC'
              OR COALESCE(m.metadata->>'human_takeover','false')='true'
              OR COALESCE(m.metadata,'{}'::jsonb) ? 'component_card')) AS source_available
        FROM memory_fact_changes f
        JOIN memory_input_origins i ON i.id=f.id AND i.user_id=f.user_id AND i.workspace_id=f.workspace_id
        JOIN chat_workspaces w ON w.id=f.workspace_id AND w.user_id=f.user_id AND w.agent_id=i.agent_id
        JOIN memories_user o ON o.id=f.old_memory_id AND o.user_id=f.user_id AND o.workspace_id=f.workspace_id
        JOIN memories_user n ON n.id=f.new_memory_id AND n.user_id=f.user_id AND n.workspace_id=f.workspace_id
        WHERE f.user_id=$1 AND f.workspace_id=$2
          AND (f.old_memory_id=ANY($3::text[]) OR f.new_memory_id=ANY($3::text[]))
          AND ($5::timestamptz IS NULL OR f.recorded_at<=$5::timestamptz)
        ORDER BY f.recorded_at DESC,f.id LIMIT $4""",
        user_id,
        workspace_id,
        memory_ids,
        limit + 1,
        known_at.isoformat() if known_at else None,
    )
    return rows


async def read_changes(
    database, *, user_id, workspace_id, memory_id, limit=20, known_at=None
):
    if not 1 <= limit <= 50:
        raise ValueError("invalid_memory_change_page")
    rows = await _read_change_rows(
        database,
        user_id=user_id,
        workspace_id=workspace_id,
        memory_ids=[memory_id],
        limit=limit,
        known_at=known_at,
    )
    return {
        "items": [
            {
                **r,
                "old_version_id": str(r["old_version_id"]),
                "new_version_id": str(r["new_version_id"]),
            }
            for r in rows[:limit]
        ],
        "limit": limit,
        "truncated": len(rows) > limit,
        "state": "recorded" if rows else "not_collected",
        "sampled_at": datetime.now(timezone.utc).isoformat(),
        "complete_history": False,
    }


async def read_constraints(database, *, user_id, workspace_id, memory_ids):
    """One bounded read for the already scoped replacement-chain candidates."""
    if not memory_ids:
        return []
    rows = await _read_change_rows(
        database,
        user_id=user_id,
        workspace_id=workspace_id,
        memory_ids=list(dict.fromkeys(memory_ids)),
        limit=320,
    )
    if len(rows) > 320:
        raise ValueError("memory_change_constraints_truncated")
    return rows
