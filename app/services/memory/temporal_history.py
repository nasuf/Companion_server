"""Bounded, permission-scoped history of semantic memory revisions.

A revision is a system observation, not a claim that its contents became true
at recorded_at. Confirmation classifications do not establish a valid interval.
"""

from datetime import datetime, timezone

from app.db import db
from app.services.memory.evidence import _table
from app.services.memory.recording.temporal_sources import public_temporal_basis


async def read_history(
    *, user_id, workspace_id, side, memory_id, before=None, limit=20
):
    table = _table(side)
    if not 1 <= limit <= 50 or (
        before is not None and (not before.isdigit() or not 0 < int(before) < 2**63)
    ):
        raise ValueError("invalid_temporal_history_page")
    targets = await db.query_raw(
        f"""SELECT w.agent_id FROM {table} m
        JOIN chat_workspaces w ON w.id=m.workspace_id AND w.user_id=m.user_id
        WHERE m.id=$1 AND m.user_id=$2 AND m.workspace_id=$3""",
        memory_id,
        user_id,
        workspace_id,
    )
    if not targets:
        raise LookupError("memory_not_found_in_scope")
    rows = await db.query_raw(
        """SELECT id,content,content_version,occur_time,statement_time,is_archived,record_kind,recorded_at
        FROM memory_temporal_versions WHERE side=$1 AND memory_id=$2 AND user_id=$3 AND workspace_id=$4
          AND ($5::bigint IS NULL OR id<$5::bigint) ORDER BY id DESC LIMIT $6""",
        side,
        memory_id,
        user_id,
        workspace_id,
        int(before) if before else None,
        limit + 1,
    )
    items = []
    for row in rows[:limit]:
        # At most 6 candidates per revision, never all history. Details do not
        # expose source text or links to deleted/out-of-scope resources.
        evidence = await db.query_raw(
            """SELECT e.temporal_basis,e.source_kind,e.source_status,
            CASE WHEN e.source_kind='input' AND i.source_status='contradiction_confirmation'
              THEN i.payload_text::jsonb->'input'->>'change_type' END AS change_type
            FROM memory_evidence_links e
            LEFT JOIN memory_input_origins i ON i.id=e.source_input_id AND i.user_id=e.user_id
              AND i.workspace_id=e.workspace_id AND i.agent_id=e.agent_id
            LEFT JOIN messages m ON m.id=e.source_message_id
            LEFT JOIN conversations c ON c.id=m.conversation_id
            WHERE e.temporal_version_id=$1 AND e.user_id=$2 AND e.workspace_id=$3 AND e.agent_id=$4
              AND (e.source_kind='input' OR (e.source_kind='message' AND c.user_id=e.user_id
                AND c.workspace_id=e.workspace_id AND c.agent_id=e.agent_id AND NOT c.is_deleted
                AND encode(sha256(convert_to(m.content,'UTF8')),'hex')=e.source_version
                AND (e.source_statement_time IS NULL OR e.source_statement_time=m.created_at)
                AND (e.source_card_version IS NULL OR e.source_card_version=
                    encode(sha256(convert_to((m.metadata->'component_card')::text,'UTF8')),'hex'))))
            ORDER BY e.id LIMIT 6""",
            row["id"],
            user_id,
            workspace_id,
            targets[0]["agent_id"],
        )
        bases = [
            public_temporal_basis(e["temporal_basis"])
            for e in evidence[:5]
            if e["temporal_basis"] is not None
        ]
        changes = {
            e["change_type"]
            for e in evidence
            if e["change_type"] in {"变化", "错误", "新增"}
        }
        items.append(
            {
                "id": str(row["id"]),
                "content": row["content"][:4000],
                "content_truncated": len(row["content"]) > 4000,
                "content_version": row["content_version"],
                "recorded_at": row["recorded_at"],
                "record_kind": row["record_kind"],
                "archived": row["is_archived"],
                "statement_time": row["statement_time"],
                "event_time": row["occur_time"],
                "temporal_evidence": bases,
                "evidence_truncated": len(evidence) > 5,
                "change_type": next(iter(changes)) if len(changes) == 1 else None,
                "validity": {"status": "unknown", "from": None, "to": None},
            }
        )
    return {
        "items": items,
        "limit": limit,
        "next_cursor": str(rows[limit - 1]["id"]) if len(rows) > limit else None,
        "state": "recorded" if rows else "empty_page" if before else "not_collected",
        "sampled_at": datetime.now(timezone.utc).isoformat(),
    }
