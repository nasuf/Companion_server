"""Live, bounded source revalidation for recalled temporal assertions.

Never cache assertion availability. A current memory candidate is only an ID
anchor for its own history, not authority to read other users' or agents' rows.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import logging

from app.db import db
from app.services.memory.evidence import content_version
from app.services.memory.recording.validity import public_validity, valid_at
from app.services.memory.retrieval.context_selector import estimate_tokens
from app.services.time_domain.timestamps import parse_time

logger = logging.getLogger(__name__)
TOKEN_BUDGET = 300
MAX_CANDIDATES = 20
PER_MEMORY_LIMIT = 6


async def _load_asserted_validity_context(
    memories, *, user_id, workspace_id, query, database=None
):
    """Return prompt data plus safe diagnostics; do not mutate source candidates."""
    info = {
        "version": 1,
        "status": "not_requested",
        "candidate_count": len(memories or []),
        "assertion_count": 0,
        "linked_source_count": 0,
        "shown_count": 0,
        "truncated": False,
        "complete_history": False,
        "token_budget": TOKEN_BUDGET,
    }
    if not memories or not user_id or not workspace_id:
        info["reason"] = "no_scoped_candidates"
        return "", {}, info
    selected = []
    seen = set()
    for memory in memories[:MAX_CANDIDATES]:
        key = (memory.source, memory.id)
        if memory.id and memory.source in {"user", "ai"} and key not in seen:
            selected.append(memory)
            seen.add(key)
    if not selected:
        info["reason"] = "no_persisted_candidates"
        return "", {}, info
    historical = query.operation.type == "as_of"
    period = (
        query.expressions[0]
        if historical and query.status == "resolved" and len(query.expressions) == 1
        else None
    )
    requested = [
        {"id": m.id, "side": m.source, "version": content_version(m.text)}
        for m in selected
    ]
    try:
        rows = await asyncio.wait_for(
            (database or db).query_raw(
                """WITH RECURSIVE requested AS (SELECT * FROM jsonb_to_recordset($3::jsonb)
                AS x(id text,side text,version text)),
            roots AS (
              SELECT m.id,'user'::text AS side,m.content FROM memories_user m
              JOIN requested r ON r.id=m.id AND r.side='user'
              WHERE m.user_id=$1 AND m.workspace_id=$2 AND NOT m.is_archived
                AND encode(sha256(convert_to(m.content,'UTF8')),'hex')=r.version
              UNION ALL
              SELECT m.id,'ai'::text AS side,m.content FROM memories_ai m
              JOIN requested r ON r.id=m.id AND r.side='ai'
              WHERE m.user_id=$1 AND m.workspace_id=$2 AND NOT m.is_archived
                AND encode(sha256(convert_to(m.content,'UTF8')),'hex')=r.version
            )
            , chain AS (
              SELECT id,side,id AS root_id,ARRAY[id]::text[] AS path FROM roots
              UNION ALL
              SELECT parent.id,chain.side,chain.root_id,chain.path||parent.id
              FROM chain CROSS JOIN LATERAL (
                SELECT id,count(*) OVER() AS branches FROM (
                  SELECT q.memory_id AS id FROM memory_quality_states q
                  WHERE q.superseded_by_memory_id=chain.id AND q.memory_source=chain.side
                    AND q.user_id=$1 AND q.workspace_id=$2
                    AND EXISTS (
                      SELECT 1 FROM memories_user m WHERE chain.side='user' AND m.id=q.memory_id
                        AND m.user_id=$1 AND m.workspace_id=$2
                      UNION ALL
                      SELECT 1 FROM memories_ai m WHERE chain.side='ai' AND m.id=q.memory_id
                        AND m.user_id=$1 AND m.workspace_id=$2)
                  ORDER BY q.memory_id LIMIT 2
                ) bounded
              ) parent
              WHERE $4::boolean AND cardinality(chain.path)<8 AND parent.branches=1
                AND NOT parent.id=ANY(chain.path)
            )
            , candidates AS (
            SELECT t.root_id AS memory_id,t.side,t.id AS historical_memory_id,
              ($4::boolean AND EXISTS (SELECT 1 FROM memory_quality_states q
                 WHERE q.superseded_by_memory_id=t.id AND q.memory_source=t.side
                   AND q.user_id=$1 AND q.workspace_id=$2
                   AND (q.memory_id=ANY(t.path) OR NOT EXISTS(SELECT 1 FROM chain covered WHERE covered.id=q.memory_id
                     AND covered.side=t.side AND covered.root_id=t.root_id)))) AS chain_truncated,a.*
            FROM chain t LEFT JOIN LATERAL (
              SELECT a.id AS assertion_id,a.assertion,a.recorded_at,a.retracted_at,
                a.valid_from,a.valid_to,a.statement_at AS assertion_statement_at,
                v.id AS version_id,v.content,v.occur_time,v.statement_time,
                NULL::text AS linked_message_id,NULL::text AS linked_source_version
              FROM memory_temporal_assertions a
              JOIN memory_temporal_versions v ON v.id=a.temporal_version_id
                AND v.user_id=a.user_id AND v.workspace_id=a.workspace_id
                AND v.side=a.side AND v.memory_id=a.memory_id
              JOIN memory_evidence_links e ON e.id=a.id AND e.temporal_version_id=v.id
              JOIN messages m ON m.id=a.source_message_id AND e.source_message_id=m.id
              JOIN conversations c ON c.id=m.conversation_id AND c.user_id=a.user_id
                AND c.workspace_id=a.workspace_id AND c.agent_id=e.agent_id AND NOT c.is_deleted
              JOIN chat_workspaces w ON w.id=a.workspace_id AND w.user_id=a.user_id AND w.agent_id=c.agent_id
              WHERE a.user_id=$1 AND a.workspace_id=$2 AND a.memory_id=t.id AND a.side=t.side
                AND m.role=CASE t.side WHEN 'user' THEN 'user' ELSE 'assistant' END
                AND e.source_version=encode(sha256(convert_to(m.content,'UTF8')),'hex')
                AND e.source_statement_time=m.created_at
                AND COALESCE(m.metadata->>'human_takeover','false')<>'true'
                AND ((e.source_card_version IS NULL AND NOT COALESCE(m.metadata,'{}'::jsonb) ? 'component_card')
                   OR e.source_card_version=encode(sha256(convert_to((m.metadata->'component_card')::text,'UTF8')),'hex'))
                AND ($4::boolean OR (a.retracted_at IS NULL AND v.id=(
                  SELECT id FROM memory_temporal_versions WHERE side=t.side AND memory_id=t.id
                    AND user_id=$1 AND workspace_id=$2 ORDER BY id DESC LIMIT 1)))
              UNION ALL
              -- Old source-backed memories remain usable without manufacturing
              -- a new validity assertion or trusting the memory summary alone.
              SELECT 'linked:'||e.id AS assertion_id,NULL::jsonb AS assertion,
                e.created_at AT TIME ZONE 'UTC' AS recorded_at,NULL::timestamptz AS retracted_at,
                NULL::timestamptz AS valid_from,NULL::timestamptz AS valid_to,
                e.source_statement_time AT TIME ZONE 'UTC' AS assertion_statement_at,
                e.temporal_version_id AS version_id,m.content,NULL::timestamp AS occur_time,
                m.created_at AS statement_time,m.id AS linked_message_id,
                e.source_version AS linked_source_version
              FROM memory_evidence_links e
              JOIN messages m ON m.id=e.source_message_id
              JOIN conversations c ON c.id=m.conversation_id AND c.user_id=e.user_id
                AND c.workspace_id=e.workspace_id AND c.agent_id=e.agent_id AND NOT c.is_deleted
              JOIN chat_workspaces w ON w.id=e.workspace_id AND w.user_id=e.user_id AND w.agent_id=e.agent_id
              JOIN memories_user target ON target.id=e.memory_id AND target.user_id=e.user_id
                AND target.workspace_id=e.workspace_id
              WHERE $4::boolean AND t.side='user' AND e.memory_source='user'
                AND e.memory_id=t.id AND e.user_id=$1 AND e.workspace_id=$2
                AND e.source_kind='message' AND e.relation='extracted_from'
                AND COALESCE(e.temporal_basis->>'source_role','fact')='fact'
                AND e.content_version=encode(sha256(convert_to(target.content,'UTF8')),'hex')
                AND m.role='user'
                AND e.source_version=encode(sha256(convert_to(m.content,'UTF8')),'hex')
                AND e.source_statement_time=m.created_at
                AND char_length(m.content) BETWEEN 1 AND 800
                AND COALESCE(m.metadata->>'human_takeover','false')<>'true'
                AND NOT COALESCE(m.metadata,'{}'::jsonb) ? 'component_card'
                AND NOT EXISTS (SELECT 1 FROM memory_temporal_assertions materialized WHERE materialized.id=e.id)
              ORDER BY assertion_statement_at DESC,recorded_at DESC,assertion_id LIMIT $5
            ) a ON TRUE
            ), ranked AS (
              SELECT *,row_number() OVER (
                PARTITION BY side,memory_id ORDER BY
                  (COALESCE(valid_from<=$6::timestamptz AND valid_to>$6::timestamptz,false)) DESC,
                  assertion_statement_at DESC NULLS LAST,recorded_at DESC NULLS LAST,assertion_id DESC
              ) AS source_rank,
              count(assertion_id) OVER (PARTITION BY side,memory_id) AS candidate_total,
              bool_or(chain_truncated) OVER (PARTITION BY side,memory_id) AS any_chain_truncated
              FROM candidates
            ) SELECT * FROM ranked WHERE source_rank<=$5""",
                user_id,
                workspace_id,
                json.dumps(requested),
                historical,
                PER_MEMORY_LIMIT + 1,
                period.start.isoformat() if period else None,
            ),
            timeout=3,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        info.update(status="unavailable", reason="source_revalidation_failed")
        logger.warning("[TEMPORAL-VALIDITY] unavailable: %s", type(exc).__name__)
        return "", {}, info
    groups = {}
    for row in rows:
        info["truncated"] |= bool(
            row.get("chain_truncated") or row.get("any_chain_truncated")
        )
        info["truncated"] |= (row.get("candidate_total") or 0) > PER_MEMORY_LIMIT
        if row.get("assertion_id") is not None:
            key = (row["side"], row["memory_id"])
            groups.setdefault(key, []).append(row)
    info.update(
        status="recorded" if groups else "not_collected", reason="scoped_live_sources"
    )

    def priority(row):
        value = public_validity(row["assertion"])
        relevant = bool(period and valid_at(value, period.start) == "inside")
        return (
            relevant,
            parse_time(value.get("statement_at"))
            or datetime.min.replace(tzinfo=timezone.utc),
            str(row["recorded_at"]),
            row["assertion_id"],
        )

    lines, associations = [], {}
    used = 0
    for memory in selected:
        key = (memory.source, memory.id)
        candidates = sorted(groups.get(key, []), key=priority, reverse=True)
        info["truncated"] |= len(candidates) > PER_MEMORY_LIMIT
        for row in candidates[:PER_MEMORY_LIMIT]:
            if row.get("linked_message_id"):
                associations[("linked", row["assertion_id"])] = [
                    {
                        "assertion_id": row["assertion_id"],
                        "version_id": row["linked_source_version"],
                        "retracted_at": None,
                        "source": "user",
                        "evidence_type": "linked_statement",
                        "content": row["content"],
                        "source_excerpt": row["content"],
                        "statement_time": row["statement_time"],
                    }
                ]
                info["linked_source_count"] += 1
                continue
            value = public_validity(row["assertion"])
            excerpt = row["assertion"].get("source_quote")
            if (
                value["state"] != "asserted"
                or not isinstance(excerpt, str)
                or not 1 <= len(excerpt) <= 500
            ):
                continue
            info["assertion_count"] += 1
            membership = "not_requested"
            if period:
                from app.services.time_domain.timestamps import exclusive_end

                start, end = period.start, exclusive_end(
                    period.end, period.end_boundary
                )
                tests = [
                    valid_at(value, start),
                    valid_at(value, end - timedelta(microseconds=1)),
                ]
                membership = (
                    "inside" if all(t == "inside" for t in tests) else "unknown"
                )
                if (
                    value["valid_to"]
                    and parse_time(value["valid_to"]) <= start
                    or value["valid_from"]
                    and parse_time(value["valid_from"]) >= end
                ):
                    membership = "outside"
            record = {
                "assertion_id": row["assertion_id"],
                "source": memory.source,
                "content": row["content"],
                "source_excerpt": excerpt,
                "validity": value,
                "membership": membership,
                "recorded_at": row["recorded_at"],
                "retracted_at": row["retracted_at"],
                "version_id": str(row["version_id"]),
                "historical_memory_id": row["historical_memory_id"],
                "event_time": row["occur_time"],
                "statement_time": row["statement_time"],
            }
            associations.setdefault(key, []).append(record)
            # Compact data only, not a new unmanaged prompt. Existing temporal
            # instructions describe uncertainty and ownership for all callers.
            item = {
                "主体": "用户" if memory.source == "user" else "角色",
                "来源原话": excerpt,
                "类型": value["kind"],
                "陈述状态": value["status"],
                "有效起": value["valid_from"],
                "有效止不含": value["valid_to"],
                "说出时间": value["statement_at"],
                "历史时段匹配": membership,
                "独立核实": False,
            }
            line = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            cost = estimate_tokens(line)
            if used + cost <= TOKEN_BUDGET:
                lines.append(line)
                used += cost
                info["shown_count"] += 1
            else:
                info["truncated"] = True
    info["truncated"] |= len(memories) > MAX_CANDIDATES
    info["estimated_tokens"] = used
    return "\n".join(lines), associations, info


async def load_validity_context(
    memories, *, user_id, workspace_id, query, database=None, recent_messages=None
):
    """Combine materialized assertions with still-live recent user statements.

    Recent plaintext is evidence even before background extraction finishes.
    It is not relabelled as an asserted interval. Only IDs and exact versions
    already loaded for this turn can be read; no additional history search.
    """
    text, records, info = await _load_asserted_validity_context(
        memories,
        user_id=user_id,
        workspace_id=workspace_id,
        query=query,
        database=database,
    )
    info["recent_source_count"] = 0
    if (
        not user_id
        or not workspace_id
        or query.operation.type != "as_of"
        or not recent_messages
        or info["status"] == "unavailable"
    ):
        return text, records, info
    requested, used, seen = [], 0, set()
    for message in reversed(recent_messages[-120:]):
        mid, content = message.get("id"), message.get("content")
        stamp = parse_time(message.get("createdAt"))
        if (
            message.get("role") != "user"
            or not isinstance(mid, str)
            or not mid
            or mid in seen
            or not isinstance(content, str)
            or not 1 <= len(content) <= 800
            or stamp is None
        ):
            continue
        if len(requested) >= 12 or used + len(content) > 4000:
            info["truncated"] = True
            break
        requested.append(
            {
                "id": mid,
                "version": content_version(content),
                "statement_at": stamp.isoformat(),
            }
        )
        seen.add(mid)
        used += len(content)
    if not requested:
        return text, records, info
    try:
        rows = await asyncio.wait_for(
            (database or db).query_raw(
                """WITH requested AS (SELECT * FROM jsonb_to_recordset($3::jsonb)
               AS x(id text,version text,statement_at timestamptz))
            SELECT m.id AS source_message_id,m.content,m.created_at AS statement_time,
              r.version AS source_version
            FROM requested r JOIN messages m ON m.id=r.id
            JOIN conversations c ON c.id=m.conversation_id
            JOIN chat_workspaces w ON w.id=c.workspace_id AND w.user_id=c.user_id
              AND w.agent_id=c.agent_id
            WHERE c.user_id=$1 AND c.workspace_id=$2 AND NOT c.is_deleted
              AND m.role='user' AND m.created_at AT TIME ZONE 'UTC'=r.statement_at
              AND encode(sha256(convert_to(m.content,'UTF8')),'hex')=r.version
              AND COALESCE(m.metadata->>'human_takeover','false')<>'true'
              AND NOT COALESCE(m.metadata,'{}'::jsonb) ? 'component_card'
            ORDER BY m.created_at,m.id LIMIT 12""",
                user_id,
                workspace_id,
                json.dumps(requested),
            ),
            timeout=3,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        info.update(status="unavailable", reason="recent_source_revalidation_failed")
        logger.warning(
            "[TEMPORAL-VALIDITY] recent source unavailable: %s", type(exc).__name__
        )
        return "", {}, info
    for row in rows:
        mid = row["source_message_id"]
        records[("dialogue", mid)] = [
            {
                "assertion_id": "message:" + mid,
                "version_id": row["source_version"],
                "retracted_at": None,
                "source": "user",
                "evidence_type": "recent_statement",
                "content": row["content"],
                "source_excerpt": row["content"],
                "statement_time": row["statement_time"],
            }
        ]
    info["recent_source_count"] = len(rows)
    if rows:
        info.update(status="recorded", reason="scoped_live_sources")
    return text, records, info


async def assertions_still_available(
    records,
    memories,
    *,
    user_id,
    workspace_id,
    query,
    database=None,
    recent_messages=None,
):
    """Recheck after model work, before generated text is handed to delivery.

    New unrelated assertions need not invalidate a reply. Losing/changing any
    supplied assertion does. This is not a lock over the eventual network send.
    """
    if not records:
        return True
    _, fresh, info = await load_validity_context(
        memories,
        user_id=user_id,
        workspace_id=workspace_id,
        query=query,
        database=database,
        recent_messages=recent_messages,
    )
    if info["status"] == "unavailable":
        return False

    def identities(groups):
        return {
            (
                key,
                item.get("assertion_id"),
                item.get("version_id"),
                str(item.get("retracted_at")),
            )
            for key, values in groups.items()
            for item in values
        }

    return identities(records).issubset(identities(fresh))
