"""Bounded source-backed topic observations; never infer event completion."""

from __future__ import annotations
from app.db import db
from app.services.runtime.execution_scope import (
    bind_conversation_scope,
    scoped_transaction,
)
from app.services.time_domain.timestamps import parse_time
from app.services.time_domain.topic_policy import MAX_TOPICS
import hashlib
import logging
import json

logger = logging.getLogger(__name__)


async def observe_messages(
    *, user_id: str, conversation_id: str, message_ids: set[str], database=None
) -> int:
    """Count only source-backed explicit topic mentions/questions, once each.

    A mention after a question records a response, not completion of the real
    event. Semantic answer/repeated-question reasoning belongs to TIME06.
    """
    from app.services.time_domain.topic_policy import topic_matches
    from app.services.time_domain.followup import _is_question, _SENTENCE

    if not message_ids or len(message_ids) > 8:
        return 0
    database = database if database is not None else db
    scope = await bind_conversation_scope(
        actor_user_id=user_id, conversation_id=conversation_id, database=database
    )
    if scope.owner_user_id != user_id:
        raise ValueError("topic_owner_mismatch")
    count = 0
    affected = set()
    async with scoped_transaction(scope, database=database) as tx:
        await tx.execute_raw(
            "SELECT pg_advisory_xact_lock(hashtextextended('topic-policy:'||$1,0))",
            scope.workspace_id,
        )
        topics = await tx.query_raw(
            """SELECT id,label FROM conversation_topics
            WHERE workspace_id=$1 AND workspace_generation=$2 AND owner_generation=$3 AND agent_generation=$4
              AND scope='topic' ORDER BY updated_at DESC,id LIMIT $5""",
            scope.workspace_id,
            scope.workspace_generation,
            scope.owner_generation,
            scope.agent_generation,
            MAX_TOPICS,
        )
        if not topics:
            return 0
        messages = await tx.query_raw(
            """SELECT id,role,content,created_at FROM messages
            WHERE conversation_id=$1 AND id=ANY($2::text[]) AND role IN ('user','assistant')
              AND COALESCE(metadata->>'human_takeover','false')<>'true'
              AND NOT COALESCE(metadata,'{}'::jsonb) ? 'component_card'
              AND created_at AT TIME ZONE 'UTC'>clock_timestamp()-interval '30 days'
            ORDER BY created_at,id FOR SHARE""",
            conversation_id,
            sorted(message_ids),
        )
        candidates = []
        for message in messages:
            stamp = parse_time(message["created_at"])
            kind = "mention" if message["role"] == "user" else "question"
            if kind == "question" and not any(
                _is_question(x) for x in _SENTENCE.findall(message["content"])
            ):
                continue
            for topic in topics:
                if topic_matches(topic["label"], message["content"]):
                    candidates.append(
                        dict(
                            topic_id=topic["id"],
                            source_message_id=message["id"],
                            user_id=user_id,
                            kind=kind,
                            statement_at=stamp.isoformat(),
                            source_version=sha256_text(message["content"]),
                        )
                    )
        if not candidates:
            return 0
        # One statement for all bounded matches; row triggers still validate
        # each source under the same ownership transaction and workspace lock.
        inserted = await tx.query_raw(
            """INSERT INTO conversation_topic_observations
            (topic_id,source_message_id,user_id,kind,statement_at,source_version)
            SELECT x.topic_id,x.source_message_id,x.user_id,x.kind,x.statement_at,x.source_version
            FROM jsonb_to_recordset($1::jsonb) AS x(topic_id text,source_message_id text,
                user_id text,kind text,statement_at timestamptz,source_version text)
            ON CONFLICT DO NOTHING RETURNING topic_id""",
            json.dumps(candidates),
        )
        count = len(inserted)
        affected = {row["topic_id"] for row in inserted}
        from app.services.time_domain.topic_retention import prune_topic_history

        await prune_topic_history(tx, sorted(affected))
        if affected:
            await tx.execute_raw(
                """UPDATE conversation_topics t SET
              asked_count=observed.questions,
              last_agent_question=observed.question_at,
              last_user_mention=observed.mention_at,
              answered_at=CASE WHEN observed.mention_at>observed.question_at THEN observed.mention_at END,
              state=CASE WHEN t.action IN ('pause','stop') AND (t.expires_at IS NULL OR t.expires_at>clock_timestamp()) THEN t.state
                WHEN observed.question_at>COALESCE(observed.mention_at,'-infinity') THEN 'awaiting_user' ELSE 'open' END
              FROM (SELECT topic_id,count(*) FILTER(WHERE kind='question')::int AS questions,
                max(statement_at) FILTER(WHERE kind='question') AS question_at,
                max(statement_at) FILTER(WHERE kind='mention') AS mention_at
                FROM conversation_topic_observations WHERE topic_id=ANY($1::text[]) GROUP BY topic_id) observed
              WHERE t.id=observed.topic_id""",
                sorted(affected),
            )
    return count


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


async def observe_reply_messages(
    conversation_id: str, message_ids: set[str], *, database=None
) -> None:
    """Best-effort bounded observation; policy enforcement never depends on it."""
    database = database if database is not None else db
    try:
        rows = await database.query_raw(
            "SELECT user_id FROM conversations WHERE id=$1 AND NOT is_deleted",
            conversation_id,
        )
        if rows:
            await observe_messages(
                user_id=rows[0]["user_id"],
                conversation_id=conversation_id,
                message_ids=message_ids,
                database=database,
            )
    except Exception as exc:
        logger.warning(
            "[TOPIC-POLICY] reply observation unavailable error=%s", type(exc).__name__
        )
