"""Audited, bounded administrator topic preference operations."""

from __future__ import annotations
from datetime import datetime, timezone
from app.db import db
from app.services.runtime.execution_scope import (
    bind_conversation_scope,
    scoped_transaction,
)
from app.services.time_domain.timestamps import parse_time
from app.services.time_domain.topic_policy import POLICY_VERSION
import hashlib
from app.services.time_domain.topic_retention import (
    OBSERVATION_JOIN,
    apply_observation_window,
    identity_capacity,
)


async def admin_page(
    *,
    user_id: str,
    workspace_id: str,
    limit: int = 20,
    after_id: str = "",
    database=None,
) -> dict:
    if not 1 <= limit <= 50 or len(after_id) > 64:
        raise ValueError("invalid_topic_page")
    database = database if database is not None else db
    scope = await database.query_raw(
        """SELECT w.id FROM chat_workspaces w
      JOIN users u ON u.id=w.user_id JOIN ai_agents a ON a.id=w.agent_id AND a.user_id=u.id
      WHERE w.id=$1 AND u.id=$2 AND u.status='active' AND w.status='active' AND a.status='active'
        AND u.archived_at IS NULL AND w.archived_at IS NULL AND a.archived_at IS NULL""",
        workspace_id,
        user_id,
    )
    if not scope:
        raise LookupError("topic_scope_unavailable")
    rows = await database.query_raw(
        """SELECT t.*,observed.*,m.conversation_id,
       (m.role='user' AND t.source_version=encode(sha256(convert_to(m.content,'UTF8')),'hex')
        AND m.created_at AT TIME ZONE 'UTC'=t.source_statement_at
        AND c.user_id=t.user_id AND c.workspace_id=t.workspace_id AND c.agent_id=t.agent_id
        AND NOT c.is_deleted AND c.archived_at IS NULL
        AND COALESCE(m.metadata->>'human_takeover','false')<>'true'
        AND NOT COALESCE(m.metadata,'{}'::jsonb) ? 'component_card') AS source_available
      FROM conversation_topics t JOIN chat_workspaces w ON w.id=t.workspace_id
      JOIN users u ON u.id=t.user_id JOIN ai_agents a ON a.id=t.agent_id
"""
        + OBSERVATION_JOIN.format(reference="clock_timestamp()")
        + """
      JOIN messages m ON m.id=t.source_message_id JOIN conversations c ON c.id=m.conversation_id
      WHERE t.user_id=$1 AND t.workspace_id=$2 AND t.id>$3
       AND t.workspace_generation=w.execution_generation AND t.owner_generation=u.execution_generation
       AND t.agent_generation=a.execution_generation
      ORDER BY t.id LIMIT $4""",
        user_id,
        workspace_id,
        after_id,
        limit + 1,
    )
    history = (
        await database.query_raw(
            """SELECT h.*, COALESCE(NULLIF(u.display_name,''),u.username) AS actor_name
        FROM unnest($1::text[]) AS selected(id)
        CROSS JOIN LATERAL (SELECT e.id,e.topic_id,e.action,e.reason,e.source_type,e.actor_user_id,e.recorded_at
          FROM conversation_topic_events e JOIN conversation_topics t ON t.id=e.topic_id
          WHERE e.topic_id=selected.id AND t.user_id=$2 AND t.workspace_id=$3
          ORDER BY e.recorded_at DESC,e.id DESC LIMIT 5) h
        LEFT JOIN users u ON u.id=h.actor_user_id""",
            [row["id"] for row in rows[:limit]],
            user_id,
            workspace_id,
        )
        if rows
        else []
    )
    now = datetime.now(timezone.utc)
    items = []
    for row in rows[:limit]:
        row = apply_observation_window(row)
        expiry = parse_time(row["expires_at"])
        item = {
            k: row[k]
            for k in (
                "id",
                "label",
                "scope",
                "state",
                "action",
                "revision",
                "source_message_id",
                "source_version",
                "source_available",
                "conversation_id",
                "policy_version",
                "asked_count",
            )
        }
        for key in (
            "source_statement_at",
            "decision_at",
            "expires_at",
            "last_user_mention",
            "last_agent_question",
            "answered_at",
            "updated_at",
        ):
            value = parse_time(row.get(key))
            item[key] = value.isoformat() if value else None
        item["effective"] = (
            "source_unavailable"
            if not row["source_available"]
            else (
                "expired"
                if expiry and expiry <= now
                else "active" if row["action"] != "resume" else "reopened"
            )
        )
        item["history"] = [
            {
                "id": e["id"],
                "action": e["action"],
                "reason": e["reason"],
                "source_type": e["source_type"],
                "actor_name": e["actor_name"] if e["source_type"] == "admin" else None,
                "recorded_at": parse_time(e["recorded_at"]).isoformat(),
            }
            for e in history
            if e["topic_id"] == row["id"]
        ]
        items.append(item)
    return {
        "items": items,
        "next_cursor": items[-1]["id"] if len(rows) > limit else None,
        "limit": limit,
        "sampled_at": now.isoformat(),
        "state": "recorded" if items else "no_recorded_topics",
        "definition_version": POLICY_VERSION,
        "complete_history": False,
        "observation_window_days": 30,
        "observation_limit_per_topic": 64,
        "audit_retained_per_topic": 128,
        "identity_capacity": await identity_capacity(
            database, user_id=user_id, workspace_id=workspace_id
        ),
    }


async def admin_correct(
    *,
    actor_id: str,
    user_id: str,
    workspace_id: str,
    ident: str,
    expected_revision: int,
    action: str,
    reason: str,
    database=None,
) -> dict:
    """An audited correction changes policy, not a user's original statement."""
    from uuid import uuid4

    if (
        action not in {"resume", "pause", "stop"}
        or not 1 <= len(reason.strip()) <= 500
        or expected_revision < 1
    ):
        raise ValueError("invalid_topic_correction")
    database = database if database is not None else db
    rows = await database.query_raw(
        """SELECT m.conversation_id FROM conversation_topics t
        JOIN messages m ON m.id=t.source_message_id
        WHERE t.id=$1 AND t.user_id=$2 AND t.workspace_id=$3""",
        ident,
        user_id,
        workspace_id,
    )
    if not rows:
        raise LookupError("topic_scope_unavailable")
    scope = await bind_conversation_scope(
        actor_user_id=actor_id,
        conversation_id=rows[0]["conversation_id"],
        database=database,
    )
    if scope.owner_user_id != user_id or scope.workspace_id != workspace_id:
        raise LookupError("topic_scope_unavailable")
    async with scoped_transaction(scope, database=database) as tx:
        admins = await tx.query_raw(
            "SELECT id FROM users WHERE id=$1 AND role='admin' AND status='active' FOR SHARE",
            actor_id,
        )
        if not admins:
            raise LookupError("topic_admin_unavailable")
        await tx.execute_raw(
            "SELECT pg_advisory_xact_lock(hashtextextended('topic-policy:'||$1,0))",
            workspace_id,
        )
        rows = await tx.query_raw(
            """SELECT t.*,m.content,m.created_at FROM conversation_topics t
            JOIN messages m ON m.id=t.source_message_id
            WHERE t.id=$1 AND t.user_id=$2 AND t.workspace_id=$3 AND t.workspace_generation=$4
              AND t.agent_generation=$5 AND t.owner_generation=$6 FOR UPDATE OF t FOR SHARE OF m""",
            ident,
            user_id,
            workspace_id,
            scope.workspace_generation,
            scope.agent_generation,
            scope.owner_generation,
        )
        if not rows:
            raise LookupError("topic_scope_unavailable")
        row = rows[0]
        if row["scope"] == "session" and action == "stop":
            raise ValueError("session_scope_cannot_become_permanent")
        if row["revision"] != expected_revision:
            raise ValueError("topic_revision_conflict")
        now = datetime.now(timezone.utc)
        from datetime import timedelta

        expiry = now + timedelta(seconds=10800) if action == "pause" else None
        version = hashlib.sha256(row["content"].encode()).hexdigest()
        statement = parse_time(row["created_at"])
        await tx.execute_raw(
            """UPDATE conversation_topics SET action=$2,state=$3,expires_at=$4::timestamptz,
            source_version=$5,source_statement_at=$6::timestamptz,decision_at=$7::timestamptz,
            revision=revision+1,updated_at=$7::timestamptz WHERE id=$1""",
            ident,
            action,
            (
                "open"
                if action == "resume"
                else "closed" if action == "stop" else "paused"
            ),
            expiry.isoformat() if expiry else None,
            version,
            statement.isoformat(),
            now.isoformat(),
        )
        event_id = uuid4().hex
        await tx.execute_raw(
            """INSERT INTO conversation_topic_events
            (id,topic_id,user_id,source_message_id,actor_user_id,source_type,action,expires_at,source_version,statement_at,ordinal,reason,policy_version)
            VALUES ($1,$2,$3,$4,$5,'admin',$6,$7::timestamptz,$8,$9::timestamptz,0,$10,$11)""",
            event_id,
            ident,
            user_id,
            row["source_message_id"],
            actor_id,
            action,
            expiry.isoformat() if expiry else None,
            version,
            statement.isoformat(),
            reason.strip(),
            POLICY_VERSION,
        )
        from app.services.time_domain.topic_retention import prune_topic_history

        await prune_topic_history(tx, [ident])
    return {
        "id": ident,
        "revision": expected_revision + 1,
        "action": action,
        "audit_event_id": event_id,
    }
