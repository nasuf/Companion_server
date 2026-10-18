"""Read-only provenance detail and cursor-limited coverage preview."""
from __future__ import annotations

from base64 import urlsafe_b64decode, urlsafe_b64encode
from datetime import datetime, timezone
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.db import db
from app.services.memory.evidence import _table, content_version
from app.services.memory.legacy_evidence_read import legacy_evidence_preview


def daily_summary_details(value):
    """Only a fixed non-content projection of the private snapshot reaches admins."""
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("timezone"), str)
        or len(value["timezone"]) > 64
    ):
        return None
    if not isinstance(value.get("local_date"), str) or not re.fullmatch(
        r"20[0-9]{2}-[0-9]{2}-[0-9]{2}", value["local_date"]
    ):
        return None
    try:
        ZoneInfo(value["timezone"])
        datetime.strptime(value["local_date"], "%Y-%m-%d")
        if datetime.fromisoformat(value["sampled_at"]).tzinfo is None:
            return None
    except (ValueError, TypeError, KeyError, ZoneInfoNotFoundError):
        return None
    bounds = {
        "schedule_slots": 96,
        "adjustments": 100,
        "proactive_messages": 5,
        "games": 200,
    }
    if any(
        type(value.get(k)) is not int or not 0 <= value[k] <= limit
        for k, limit in bounds.items()
    ):
        return None
    if type(value.get("proactive_truncated")) is not bool:
        return None
    return {
        key: value[key]
        for key in (
            *bounds,
            "local_date",
            "timezone",
            "sampled_at",
            "proactive_truncated",
        )
    }


def consolidation_details(value):
    """A bounded public projection; private parents/prompts never leave storage."""
    if (not isinstance(value, dict) or value.get("side") not in ("user", "ai")
            or type(value.get("parents")) is not int or not 5 <= value["parents"] <= 300):
        return None
    try:
        if datetime.fromisoformat(value["sampled_at"]).tzinfo is None:
            return None
    except (ValueError, TypeError, KeyError):
        return None
    return {key: value[key] for key in ("side", "parents", "sampled_at")}


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
             WHEN e.source_kind='input' THEN
               CASE WHEN i.id IS NULL THEN 'deleted'
                    WHEN i.user_id IS DISTINCT FROM e.user_id OR i.workspace_id IS DISTINCT FROM e.workspace_id
                         OR i.agent_id IS DISTINCT FROM e.agent_id THEN 'unavailable'
                    WHEN i.source_version IS DISTINCT FROM e.source_version THEN 'changed'
                    ELSE 'recorded_snapshot' END
             WHEN e.source_kind='profile' THEN
               CASE WHEN o.id IS NULL THEN 'deleted'
                    WHEN o.user_id IS DISTINCT FROM e.user_id OR o.workspace_id IS DISTINCT FROM e.workspace_id
                         OR o.agent_id IS DISTINCT FROM e.agent_id THEN 'unavailable'
                    WHEN o.source_version IS DISTINCT FROM e.source_version THEN 'changed'
                    ELSE 'recorded_snapshot' END
             WHEN e.source_kind='message' THEN
               CASE WHEN e.source_message_id IS NULL THEN 'deleted'
                    WHEN c.user_id IS DISTINCT FROM e.user_id OR c.workspace_id IS DISTINCT FROM e.workspace_id
                         OR c.agent_id IS DISTINCT FROM e.agent_id OR c.is_deleted THEN 'unavailable'
                    WHEN encode(sha256(convert_to(m.content,'UTF8')),'hex')<>e.source_version THEN 'changed'
                    WHEN e.source_card_version IS NOT NULL AND e.source_card_version IS DISTINCT FROM
                         encode(sha256(convert_to((m.metadata->'component_card')::text,'UTF8')),'hex') THEN 'changed'
                    ELSE 'available' END
             WHEN e.source_kind='memory' THEN
               CASE WHEN p.id IS NULL THEN 'deleted'
                    WHEN p.user_id IS DISTINCT FROM e.source_user_id
                         OR p.workspace_id IS DISTINCT FROM e.source_workspace_id THEN 'unavailable'
                    WHEN e.relation='template_copy' AND p.workspace_id<>e.workspace_id
                         AND NOT EXISTS (SELECT 1 FROM ai_agents a JOIN chat_workspaces w ON w.agent_id=a.source_template_id
                           JOIN ai_agents t ON t.id=w.agent_id AND t.user_id=w.user_id
                           WHERE a.id=e.agent_id AND w.id=p.workspace_id AND w.user_id=p.user_id) THEN 'unavailable'
                    WHEN encode(sha256(convert_to(p.content,'UTF8')),'hex')<>e.source_version THEN 'changed'
                    ELSE 'available' END
             WHEN e.source_kind IN ('unlinked','import') THEN 'unverified'
             ELSE 'recorded_snapshot' END AS availability,
             c.id AS conversation_id, o.input_status AS profile_input_status,
             o.format_version AS profile_format_version,
             i.source_status AS input_kind, i.format_version AS input_format_version,
             CASE WHEN i.source_status='admin_repair' THEN i.payload_text::jsonb->'input'->>'action' END AS repair_action,
             CASE WHEN i.source_status='contradiction_confirmation'
               THEN i.payload_text::jsonb->'input'->>'change_type' END AS confirmation_change,
             CASE WHEN i.source_status='daily_summary' THEN i.payload_text::jsonb->'input'->'summary' END AS daily_summary,
             CASE WHEN i.source_status='consolidation' THEN i.payload_text::jsonb->'input'->'summary' END AS consolidation,
             p.is_archived AS parent_archived
           FROM memory_evidence_links e
           LEFT JOIN messages m ON m.id=e.source_message_id
           LEFT JOIN conversations c ON c.id=m.conversation_id
           LEFT JOIN memory_profile_origins o ON o.id=e.source_profile_id
           LEFT JOIN memory_input_origins i ON i.id=e.source_input_id
           LEFT JOIN LATERAL (
             SELECT id,user_id,workspace_id,content,is_archived FROM memories_user WHERE id=e.parent_user_memory_id
             UNION ALL
             SELECT id,user_id,workspace_id,content,is_archived FROM memories_ai WHERE id=e.parent_ai_memory_id
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
            "conversation_id":row["conversation_id"] if accessible and row["source_kind"]=="message" else None,
            "source_archived":row["parent_archived"] if accessible and row["source_kind"]=="memory" else None,
            "repair_action":row["repair_action"] if accessible and row["source_kind"]=="input" and row["repair_action"] in {"merge_memories","insert_replacement_memory","archive_memory","downgrade_memory","mark_verified"} else None,
            "consolidation": consolidation_details(row.get("consolidation")) if accessible and row["source_kind"]=="input" else None,
            "daily_summary": daily_summary_details(row["daily_summary"]) if accessible and row["source_kind"]=="input" else None,
            "confirmation_change": row["confirmation_change"] if accessible and row["source_kind"]=="input"
                and row["confirmation_change"] in {"变化","错误","新增"} else None,
            "profile": {"input_status":row["profile_input_status"], "format_version":row["profile_format_version"]}
                if accessible and row["source_kind"]=="profile" else None,
            "input": {"kind":row["input_kind"], "format_version":row["input_format_version"]}
                if accessible and row["source_kind"]=="input" else None})
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
