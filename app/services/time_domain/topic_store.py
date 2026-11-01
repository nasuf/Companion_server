"""Durable source-bound topic preferences; all writes use current SQL ownership.

No Redis TTL, in-process cache, historical inference, or model-supplied owner can
make a preference effective. Policy event history is audit-only: readers use the
current projection and never replay an older event after source deletion.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import logging

from app.db import db
from app.services.runtime.execution_scope import (
    bind_conversation_scope,
    scoped_transaction,
)
from app.services.time_domain.timestamps import parse_time
from app.services.time_domain.topic_policy import (
    MAX_TOPICS,
    MAX_SOURCE_IDS,
    POLICY_VERSION,
    TopicRestriction,
    parse_commands,
    topic_id,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TopicSnapshot:
    status: str
    items: tuple[TopicRestriction, ...] = ()
    truncated: bool = False
    sampled_at: datetime | None = None

    def diagnostics(self):
        return {
            "status": self.status,
            "policy_version": POLICY_VERSION,
            "sampled_at": self.sampled_at.isoformat() if self.sampled_at else None,
            "truncated": self.truncated,
            "limit": MAX_TOPICS,
            "items": [x.diagnostic() for x in self.items[:20]],
            "item_count": len(self.items),
            "display_limit": 20,
            "complete_history": False,
            "observation_window_days": 30,
            "observation_limit_per_topic": 64,
        }


async def record_commands(
    *, user_id: str, conversation_id: str, message_ids: set[str], database=None
) -> int:
    """Reparse actual persisted messages inside the source/erasure fence.

    No caller text is used as evidence; aggregated messages keep their individual
    statement clocks. Duplicate delivery cannot reset expiry or append events.
    """
    if not message_ids or len(message_ids) > MAX_SOURCE_IDS:
        raise ValueError("topic_source_ids_required_or_over_limit")
    database = database if database is not None else db
    scope = await bind_conversation_scope(
        actor_user_id=user_id, conversation_id=conversation_id, database=database
    )
    if scope.owner_user_id != user_id:
        raise ValueError("topic_owner_mismatch")
    inserted = 0
    affected = set()
    async with scoped_transaction(scope, database=database) as tx:
        await tx.execute_raw(
            "SELECT pg_advisory_xact_lock(hashtextextended('topic-policy:'||$1,0))",
            scope.workspace_id,
        )
        rows = await tx.query_raw(
            """SELECT id,content,created_at FROM messages
            WHERE conversation_id=$1 AND id=ANY($2::text[]) AND role='user'
              AND COALESCE(metadata->>'human_takeover','false')<>'true'
              AND NOT COALESCE(metadata,'{}'::jsonb) ? 'component_card'
            ORDER BY created_at,id FOR SHARE""",
            conversation_id,
            sorted(message_ids),
        )
        if len(rows) != len(message_ids):
            raise ValueError("topic_source_unavailable")
        for row in rows:
            text, anchor = row["content"], parse_time(row["created_at"])
            if anchor is None:
                raise ValueError("topic_source_clock_unavailable")
            version = hashlib.sha256(text.encode()).hexdigest()
            for command in parse_commands(text, statement_at=anchor):
                if command.source_quote not in text:
                    continue
                ident = topic_id(
                    scope.workspace_id, scope.workspace_generation, command.key
                )
                event_id = hashlib.sha256(
                    (ident + "\0" + row["id"] + "\0" + str(command.ordinal)).encode()
                ).hexdigest()
                # Check idempotency before changing the projection. A stale
                # replay cannot undo a later source edit or admin correction.
                prior = await tx.query_raw(
                    "SELECT id FROM conversation_topic_events WHERE id=$1", event_id
                )
                if prior:
                    continue
                existing = await tx.query_raw(
                    "SELECT id FROM conversation_topics WHERE id=$1", ident
                )
                if not existing:
                    fence = await tx.query_raw(
                        "SELECT decision_at FROM conversation_topic_fences WHERE topic_id=$1 AND deleted",
                        ident,
                    )
                    if fence and anchor <= parse_time(fence[0]["decision_at"]):
                        continue
                if command.action == "resume" and not existing:
                    continue
                state = (
                    "open"
                    if command.action == "resume"
                    else "closed" if command.action == "stop" else "paused"
                )
                result = await tx.query_raw(
                    """INSERT INTO conversation_topics
                  (id,user_id,agent_id,workspace_id,workspace_generation,agent_generation,owner_generation,
                   topic_key,label,scope,state,action,expires_at,source_message_id,source_version,
                   source_statement_at,decision_at,source_ordinal,policy_version,last_user_mention)
                  VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13::timestamptz,$14,$15,$16::timestamptz,$16::timestamptz,$17,$18,$16::timestamptz)
                  ON CONFLICT (id) DO UPDATE SET
                    state=EXCLUDED.state,action=EXCLUDED.action,expires_at=EXCLUDED.expires_at,
                    source_message_id=EXCLUDED.source_message_id,source_version=EXCLUDED.source_version,
                    source_statement_at=EXCLUDED.source_statement_at,decision_at=EXCLUDED.decision_at,source_ordinal=EXCLUDED.source_ordinal,
                    policy_version=EXCLUDED.policy_version,last_user_mention=EXCLUDED.last_user_mention,
                    revision=conversation_topics.revision+1,updated_at=clock_timestamp()
                  WHERE EXCLUDED.decision_at>conversation_topics.decision_at
                    OR (EXCLUDED.decision_at=conversation_topics.decision_at AND
                      ((EXCLUDED.source_message_id=conversation_topics.source_message_id AND EXCLUDED.source_ordinal>conversation_topics.source_ordinal)
                       OR (EXCLUDED.source_message_id<>conversation_topics.source_message_id AND EXCLUDED.action IN ('pause','stop')
                           AND (conversation_topics.action='resume' OR (EXCLUDED.action='stop' AND conversation_topics.action='pause')
                             OR (EXCLUDED.action='pause' AND conversation_topics.action='pause' AND EXCLUDED.expires_at>conversation_topics.expires_at)))))
                  RETURNING id""",
                    ident,
                    user_id,
                    scope.agent_id,
                    scope.workspace_id,
                    scope.workspace_generation,
                    scope.agent_generation,
                    scope.owner_generation,
                    command.key,
                    command.topic,
                    command.scope,
                    state,
                    command.action,
                    command.expires_at.isoformat() if command.expires_at else None,
                    row["id"],
                    version,
                    anchor.isoformat(),
                    command.ordinal,
                    POLICY_VERSION,
                )
                # Audit stale requests too, but do not grant them projection authority.
                await tx.execute_raw(
                    """INSERT INTO conversation_topic_events
                    (id,topic_id,user_id,source_message_id,action,expires_at,source_version,statement_at,ordinal,reason,policy_version)
                    VALUES ($1,$2,$3,$4,$5,$6::timestamptz,$7,$8::timestamptz,$9,$10,$11)
                    ON CONFLICT (id) DO NOTHING""",
                    event_id,
                    ident,
                    user_id,
                    row["id"],
                    command.action,
                    command.expires_at.isoformat() if command.expires_at else None,
                    version,
                    anchor.isoformat(),
                    command.ordinal,
                    command.reason if result else "stale_request_not_applied",
                    POLICY_VERSION,
                )
                inserted += bool(result)
                affected.add(ident)
        from app.services.time_domain.topic_retention import (
            prune_topic_history,
            prune_inactive_topics,
        )

        # Aggregation may contain >128 distinct topics; prune in bounded chunks.
        targets = sorted(affected)
        for offset in range(0, len(targets), MAX_TOPICS):
            await prune_topic_history(tx, targets[offset : offset + MAX_TOPICS])
        await prune_inactive_topics(tx, scope.workspace_id)
    return inserted


async def record_turn_preferences(
    *,
    user_id: str,
    conversation_id: str,
    message_ids: set[str],
    current_message: str,
    statement_at: datetime,
    database=None,
) -> dict:
    """Capture explicit preferences before crisis/pending reply short circuits.

    Failure remains visible; it must not suppress a required crisis reply. The
    ordinary policy reader retries the same idempotent source write later.
    """
    if len(message_ids) <= 1 and not parse_commands(
        current_message, statement_at=statement_at
    ):
        return {"status": "not_requested"}
    try:
        count = await record_commands(
            user_id=user_id,
            conversation_id=conversation_id,
            message_ids=message_ids,
            database=database,
        )
        return {"status": "recorded", "applied_commands": count}
    except Exception as exc:
        logger.warning(
            "[TOPIC-POLICY] source write unavailable error=%s", type(exc).__name__
        )
        return {"status": "unavailable", "reason": type(exc).__name__}


from app.services.time_domain.topic_retention import (
    OBSERVATION_JOIN,
    apply_observation_window,
    identity_capacity,
)

_READ_SQL = (
    """SELECT t.*, observed.*,
  (m.id IS NOT NULL AND m.role='user'
    AND encode(sha256(convert_to(m.content,'UTF8')),'hex')=t.source_version
    AND m.created_at AT TIME ZONE 'UTC'=t.source_statement_at
    AND COALESCE(m.metadata->>'human_takeover','false')<>'true'
    AND NOT COALESCE(m.metadata,'{}'::jsonb) ? 'component_card'
    AND c.user_id=t.user_id AND c.workspace_id=t.workspace_id AND c.agent_id=t.agent_id
    AND NOT c.is_deleted AND c.archived_at IS NULL) AS source_available
FROM conversation_topics t
JOIN users u ON u.id=t.user_id AND u.execution_generation=t.owner_generation
JOIN chat_workspaces w ON w.id=t.workspace_id AND w.user_id=u.id AND w.execution_generation=t.workspace_generation
JOIN ai_agents a ON a.id=t.agent_id AND a.user_id=u.id AND a.id=w.agent_id AND a.execution_generation=t.agent_generation
"""
    + OBSERVATION_JOIN.format(reference="$3::timestamptz")
    + """
LEFT JOIN messages m ON m.id=t.source_message_id
LEFT JOIN conversations c ON c.id=m.conversation_id
WHERE t.user_id=$1 AND t.workspace_id=$2
  AND u.status='active' AND u.archived_at IS NULL AND w.status='active' AND w.archived_at IS NULL
  AND a.status='active' AND a.archived_at IS NULL
  AND (t.expires_at IS NULL OR t.expires_at>$3::timestamptz)
  AND t.action IN ('pause','stop')
ORDER BY t.updated_at DESC,t.id LIMIT $4"""
)


async def load_snapshot(
    *, user_id: str, workspace_id: str, now: datetime, database=None
) -> TopicSnapshot:
    database = database if database is not None else db
    anchor = parse_time(now)
    if anchor is None or not user_id or not workspace_id:
        return TopicSnapshot("scope_unavailable", sampled_at=anchor)
    capacity = await identity_capacity(
        database, user_id=user_id, workspace_id=workspace_id
    )
    if capacity["exhausted"]:
        return TopicSnapshot("capacity_exceeded", sampled_at=anchor, truncated=True)
    rows = await database.query_raw(
        _READ_SQL, user_id, workspace_id, anchor.isoformat(), MAX_TOPICS + 1
    )
    items = []
    invalid = 0
    for row in rows[:MAX_TOPICS]:
        row = apply_observation_window(row)
        if not row["source_available"]:
            invalid += 1
            continue
        items.append(
            TopicRestriction(
                id=row["id"],
                topic=row["label"],
                scope=row["scope"],
                action=row["action"],
                expires_at=parse_time(row["expires_at"]),
                source_id=row["source_message_id"],
                version=row["revision"],
                state=row["state"],
                last_user_mention=parse_time(row["last_user_mention"]),
                last_agent_question=parse_time(row["last_agent_question"]),
                answered_at=parse_time(row["answered_at"]),
                asked_count=row["asked_count"],
            )
        )
    status = (
        "capacity_exceeded"
        if len(rows) > MAX_TOPICS
        else (
            "source_unavailable"
            if invalid
            else "recorded" if items else "no_active_restrictions"
        )
    )
    return TopicSnapshot(status, tuple(items), len(rows) > MAX_TOPICS, anchor)


async def resolve_policy(
    policy,
    *,
    user_id: str,
    workspace_id: str,
    now: datetime,
    conversation_id: str | None = None,
    message_ids: set[str] | None = None,
    current_message: str = "",
    database=None,
):
    """Shared ordinary/A/B reader. Missing storage is never treated as consent."""
    try:
        if (
            conversation_id
            and message_ids
            and (
                len(message_ids) > 1
                or parse_commands(current_message, statement_at=now)
            )
        ):
            await record_commands(
                user_id=user_id,
                conversation_id=conversation_id,
                message_ids=message_ids,
                database=database,
            )
        snapshot = await load_snapshot(
            user_id=user_id, workspace_id=workspace_id, now=now, database=database
        )
    except Exception as exc:
        logger.warning("[TOPIC-POLICY] unavailable error=%s", type(exc).__name__)
        snapshot = TopicSnapshot("unavailable", sampled_at=now)
    unavailable = snapshot.status in {
        "unavailable",
        "scope_unavailable",
        "capacity_exceeded",
        "source_unavailable",
    }
    global_stop = any(
        item.scope in {"global", "session"} and item.active(now)
        for item in snapshot.items
    )
    return replace(
        policy,
        topic_snapshot=snapshot,
        optional_questions_allowed=policy.optional_questions_allowed
        and not unavailable
        and not global_stop,
        reason=(
            "topic_store_unavailable"
            if unavailable
            else "persisted_general_opt_out" if global_stop else policy.reason
        ),
    )


# These public entry points keep dependency injection at the shared boundary.
# Dedicated modules own administrator audit and non-authoritative observations.
async def admin_page(*, user_id, workspace_id, limit=20, after_id="", database=None):
    from app.services.time_domain.topic_admin import admin_page as read

    return await read(
        user_id=user_id,
        workspace_id=workspace_id,
        limit=limit,
        after_id=after_id,
        database=database if database is not None else db,
    )


async def admin_correct(
    *,
    actor_id,
    user_id,
    workspace_id,
    ident,
    expected_revision,
    action,
    reason,
    database=None,
):
    from app.services.time_domain.topic_admin import admin_correct as correct

    return await correct(
        actor_id=actor_id,
        user_id=user_id,
        workspace_id=workspace_id,
        ident=ident,
        expected_revision=expected_revision,
        action=action,
        reason=reason,
        database=database if database is not None else db,
    )


async def observe_messages(*, user_id, conversation_id, message_ids, database=None):
    from app.services.time_domain.topic_observations import observe_messages as observe

    return await observe(
        user_id=user_id,
        conversation_id=conversation_id,
        message_ids=message_ids,
        database=database if database is not None else db,
    )


async def observe_reply_messages(conversation_id, message_ids):
    from app.services.time_domain.topic_observations import (
        observe_reply_messages as observe,
    )

    return await observe(conversation_id, message_ids, database=db)
