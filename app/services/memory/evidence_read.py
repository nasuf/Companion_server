"""Read-only provenance detail and cursor-limited coverage preview."""
from __future__ import annotations

from base64 import urlsafe_b64decode, urlsafe_b64encode
from datetime import datetime, timezone

from app.db import db
from app.services.memory.evidence import _table, content_version
from app.services.memory.legacy_evidence_read import legacy_evidence_preview


def _cursor(value: str | None) -> str:
    if value is None:
        return ""
    if len(value) > 300:
        raise ValueError("invalid_evidence_cursor")
    try:
        decoded = urlsafe_b64decode(value.encode()).decode()
        if len(decoded) != 64 or any(c not in "0123456789abcdef" for c in decoded):
            raise ValueError
        return decoded
    except (ValueError, UnicodeError) as exc:
        raise ValueError("invalid_evidence_cursor") from exc


async def memory_evidence_detail(*, user_id: str, workspace_id: str, side: str,
                                 memory_id: str, limit: int = 20, cursor: str | None = None) -> dict:
    if not 1 <= limit <= 50:
        raise ValueError("invalid_evidence_limit")
    after = _cursor(cursor)
    targets = await db.query_raw(
        f"""SELECT m.content,m.provenance,m.created_at,w.agent_id FROM {_table(side)} m
            JOIN chat_workspaces w ON w.id=m.workspace_id AND w.user_id=m.user_id
            WHERE m.id=$1 AND m.user_id=$2 AND m.workspace_id=$3""", memory_id,user_id,workspace_id)
    if not targets:
        raise LookupError("memory_not_found_in_scope")
    version = content_version(targets[0]["content"])
    rows = await db.query_raw(
        """SELECT e.*,
             CASE WHEN e.agent_id IS DISTINCT FROM $7::text THEN 'unavailable'
             WHEN e.source_kind='message' THEN
               CASE WHEN e.source_message_id IS NULL THEN 'deleted'
                    WHEN c.user_id IS DISTINCT FROM e.user_id OR c.workspace_id IS DISTINCT FROM e.workspace_id
                         OR c.agent_id IS DISTINCT FROM e.agent_id OR c.is_deleted THEN 'unavailable'
                    WHEN encode(sha256(convert_to(m.content,'UTF8')),'hex')<>e.source_version THEN 'changed'
                    ELSE 'available' END
             WHEN e.source_kind='memory' THEN
               CASE WHEN p.id IS NULL THEN 'deleted'
                    WHEN p.user_id IS DISTINCT FROM e.source_user_id
                         OR p.workspace_id IS DISTINCT FROM e.source_workspace_id THEN 'unavailable'
                    WHEN e.relation='template_copy' AND p.workspace_id<>e.workspace_id
                         AND NOT EXISTS (SELECT 1 FROM ai_agents a JOIN chat_workspaces w ON w.agent_id=a.source_template_id
                           WHERE a.id=e.agent_id AND w.id=p.workspace_id) THEN 'unavailable'
                    WHEN encode(sha256(convert_to(p.content,'UTF8')),'hex')<>e.source_version THEN 'changed'
                    ELSE 'available' END
             WHEN e.source_kind IN ('unlinked','import') THEN 'unverified'
             ELSE 'recorded_snapshot' END AS availability,
             c.id AS conversation_id
           FROM memory_evidence_links e
           LEFT JOIN messages m ON m.id=e.source_message_id
           LEFT JOIN conversations c ON c.id=m.conversation_id
           LEFT JOIN LATERAL (
             SELECT id,user_id,workspace_id,content FROM memories_user WHERE id=e.parent_user_memory_id
             UNION ALL
             SELECT id,user_id,workspace_id,content FROM memories_ai WHERE id=e.parent_ai_memory_id
           ) p ON true
           WHERE e.memory_id=$1 AND e.memory_source=$2 AND e.user_id=$3 AND e.workspace_id=$4 AND e.id>$5
           ORDER BY e.id LIMIT $6""", memory_id,side,user_id,workspace_id,after,limit+1,targets[0]["agent_id"])
    items = []
    for row in rows[:limit]:
        accessible = row["availability"] not in {"deleted", "unavailable", "unverified"}
        items.append({"id":row["id"],"source_kind":row["source_kind"],
            "source_ref":row["source_ref"] if accessible else None,
            "source_side":row["source_memory_side"],
            "source_role":row["source_role"],"source_status":row["source_status"],
            "source_version":row["source_version"],"content_version":row["content_version"],
            "current_content":row["content_version"]==version,
            "extractor_version":row["extractor_version"],"relation":row["relation"],
            "availability":row["availability"],"created_at":row["created_at"],
            "conversation_id":row["conversation_id"] if accessible and row["source_kind"]=="message" else None})
    # A previous-page match must not disappear as users page through history.
    counts = await db.query_raw(
        """SELECT EXISTS(SELECT 1 FROM memory_evidence_links WHERE memory_id=$1 AND memory_source=$2
           AND user_id=$3 AND workspace_id=$4 AND content_version=$5
           AND source_kind NOT IN ('unlinked','import') AND agent_id=$6) AS has_current,
           EXISTS(SELECT 1 FROM memory_evidence_links WHERE memory_id=$1 AND memory_source=$2
           AND user_id=$3 AND workspace_id=$4) AS has_history""",memory_id,side,user_id,workspace_id,version,targets[0]["agent_id"])
    legacy = await legacy_evidence_preview(database=db, user_id=user_id, workspace_id=workspace_id,
        side=side, memory_id=memory_id, agent_id=targets[0]["agent_id"],
        target_created_at=targets[0]["created_at"])
    return {"memory_id":memory_id,"memory_source":side,"user_id":user_id,"workspace_id":workspace_id,
        "agent_id":targets[0]["agent_id"],"content_version":version,"provenance":targets[0]["provenance"],
        "state":"linked" if counts[0]["has_current"] else "current_unlinked" if counts[0]["has_history"] else "historical_unknown",
        "items":items,"limit":limit,"legacy":legacy,
        "next_cursor":urlsafe_b64encode(rows[limit-1]["id"].encode()).decode() if len(rows)>limit else None,
        "sampled_at":datetime.now(timezone.utc).isoformat()}


async def audit_evidence_page(*, user_id: str, workspace_id: str, side: str,
                              after_id: str = "", limit: int = 100) -> dict:
    """One bounded page, no inference/backfill/write. Caller controls stopping."""
    if not 1 <= limit <= 500 or len(after_id)>200:
        raise ValueError("invalid_audit_page")
    scope = await db.query_raw("SELECT id FROM chat_workspaces WHERE id=$1 AND user_id=$2",
                               workspace_id,user_id)
    if not scope:
        raise LookupError("workspace_not_found_in_scope")
    rows = await db.query_raw(
        f"""SELECT m.id,EXISTS(SELECT 1 FROM memory_evidence_links e WHERE e.memory_id=m.id
             AND e.memory_source=$4 AND e.user_id=m.user_id AND e.workspace_id=m.workspace_id
             AND e.content_version=encode(sha256(convert_to(m.content,'UTF8')),'hex')
             AND e.source_kind NOT IN ('unlinked','import') AND e.agent_id=w.agent_id) AS linked
           FROM {_table(side)} m JOIN chat_workspaces w ON w.id=m.workspace_id AND w.user_id=m.user_id
           WHERE m.user_id=$1 AND m.workspace_id=$2 AND m.id>$3
           ORDER BY m.id LIMIT $5""",user_id,workspace_id,after_id,side,limit+1)
    page=rows[:limit]
    return {"dry_run":True,"memory_source":side,"user_id":user_id,"workspace_id":workspace_id,
        "checked":len(page),"linked":sum(bool(row["linked"]) for row in page),
        "unknown":sum(not row["linked"] for row in page),
        "next_after_id":page[-1]["id"] if len(rows)>limit else None,
        "denominator":"this_page_only","linked_definition":"current_content_has_recorded_origin_not_verified_truth",
        "sampled_at":datetime.now(timezone.utc).isoformat()}
