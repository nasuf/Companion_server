"""Immutable, revision-bound time assertions and bounded scoped readers."""

from __future__ import annotations

from datetime import datetime, timezone
import json

from app.services.memory.recording.validity import (
    public_validity,
    resolve_validity,
    valid_at,
)
from app.services.time_domain.timestamps import parse_time


async def persist_assertion(
    *,
    database,
    evidence_id,
    revision_id,
    source,
    origin,
    user_id,
    workspace_id,
    side,
    memory_id,
):
    basis = source.temporal_basis or {}
    raw = basis.get("validity")
    value = public_validity(raw)
    if value["state"] != "asserted":
        return
    expected_role = "user" if side == "user" else "assistant"
    if (
        source.kind != "message"
        or source.expected_role != expected_role
        or origin.get("role") != expected_role
        or basis.get("source_role") != "fact"
        or revision_id is None
        or raw.get("source_version") != origin.get("version")
        or source.expected_statement_time is None
        or parse_time(value["statement_at"]) != source.expected_statement_time
    ):
        raise ValueError("temporal_assertion_source_mismatch")
    # A frozen interpretation is not permission to alter its endpoints. Verify
    # its quoted boundary evidence again before committing the subordinate row.
    proposed = {
        "kind": raw.get("kind"),
        "status": raw.get("status"),
        "source_index": 1,
        "quote": raw.get("source_quote"),
        "valid_from_quote": raw.get("valid_from_quote"),
        "valid_to_quote": raw.get("valid_to_quote"),
    }
    verified = resolve_validity({"temporal_assertion": proposed}, [source], (1,))
    if public_validity(verified) != value:
        raise ValueError("temporal_assertion_bounds_changed")
    await database.execute_raw(
        """INSERT INTO memory_temporal_assertions
        (id,temporal_version_id,source_message_id,user_id,workspace_id,side,memory_id,
         assertion,statement_at,valid_from,valid_to)
        VALUES ($1,$2::bigint,$3,$4,$5,$6,$7,$8::jsonb,$9::timestamptz,$10::timestamptz,$11::timestamptz)
        ON CONFLICT (id) DO NOTHING""",
        evidence_id,
        revision_id,
        source.ref,
        user_id,
        workspace_id,
        side,
        memory_id,
        json.dumps(raw, ensure_ascii=False),
        value["statement_at"],
        value["valid_from"],
        value["valid_to"],
    )


async def read_assertions(
    *,
    database,
    user_id,
    workspace_id,
    side,
    memory_id,
    revision_ids: list[int],
    known_at: datetime | None = None,
    at: datetime | None = None,
    limit: int = 20,
) -> dict:
    """Read only supplied revisions, revalidating their current live origins.

    known_at is system knowledge time; at is real-world validity time. Neither
    proves completeness. Edited/deleted sources are excluded even for old dates.
    """
    if (
        side not in {"user", "ai"}
        or not 1 <= limit <= 50
        or not 1 <= len(revision_ids) <= 50
        or any(type(i) is not int or not 0 < i < 2**63 for i in revision_ids)
    ):
        raise ValueError("invalid_temporal_assertion_scope")
    for date in (known_at, at):
        if date is not None and (not isinstance(date, datetime) or date.tzinfo is None):
            raise ValueError("invalid_temporal_assertion_time")
    rows = await database.query_raw(
        """SELECT a.id,a.temporal_version_id,a.assertion,a.recorded_at,a.retracted_at,
            a.statement_at,v.content,v.content_version,v.occur_time,v.statement_time
        FROM memory_temporal_assertions a
        JOIN memory_temporal_versions v ON v.id=a.temporal_version_id
          AND v.user_id=a.user_id AND v.workspace_id=a.workspace_id AND v.side=a.side AND v.memory_id=a.memory_id
        JOIN memory_evidence_links e ON e.id=a.id AND e.temporal_version_id=v.id
        JOIN messages m ON m.id=a.source_message_id AND e.source_message_id=m.id
        JOIN conversations c ON c.id=m.conversation_id AND c.user_id=a.user_id
          AND c.workspace_id=a.workspace_id AND c.agent_id=e.agent_id AND NOT c.is_deleted
        JOIN chat_workspaces w ON w.id=a.workspace_id AND w.user_id=a.user_id AND w.agent_id=c.agent_id
        WHERE a.user_id=$1 AND a.workspace_id=$2 AND a.side=$3 AND a.memory_id=$4
          AND a.temporal_version_id=ANY($5::bigint[])
          AND m.role=CASE a.side WHEN 'user' THEN 'user' ELSE 'assistant' END
          AND e.source_version=encode(sha256(convert_to(m.content,'UTF8')),'hex')
          AND e.source_statement_time=m.created_at
          AND COALESCE(m.metadata->>'human_takeover','false')<>'true'
          AND ((e.source_card_version IS NULL AND NOT COALESCE(m.metadata,'{}'::jsonb) ? 'component_card')
            OR e.source_card_version=encode(sha256(convert_to((m.metadata->'component_card')::text,'UTF8')),'hex'))
          AND ($6::timestamptz IS NULL OR (a.recorded_at<=$6::timestamptz
             AND (a.retracted_at IS NULL OR a.retracted_at>$6::timestamptz)))
        ORDER BY a.statement_at DESC,a.recorded_at DESC,a.id LIMIT $7""",
        user_id,
        workspace_id,
        side,
        memory_id,
        revision_ids,
        known_at.isoformat() if known_at else None,
        limit + 1,
    )
    items = []
    for row in rows[:limit]:
        value = public_validity(row["assertion"])
        if value["state"] != "asserted":
            continue
        items.append(
            {
                "id": row["id"],
                "version_id": str(row["temporal_version_id"]),
                "validity": value,
                "membership": valid_at(value, at) if at else "not_requested",
                "recorded_at": row["recorded_at"],
                "retracted_at": row["retracted_at"],
                "content": row["content"][:4000],
                "content_version": row["content_version"],
                "event_time": row["occur_time"],
                "statement_time": row["statement_time"],
            }
        )
    return {
        "items": items,
        "truncated": len(rows) > limit,
        "limit": limit,
        "state": "recorded" if items else "not_collected_or_source_unavailable",
        "at": at.isoformat() if at else None,
        "known_at": known_at.isoformat() if known_at else None,
        "complete_history": False,
        "sampled_at": datetime.now(timezone.utc).isoformat(),
    }
