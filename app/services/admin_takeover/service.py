"""Human takeover records, exclusive claim, and the post-claim message gate.

SQL owns the conditional claim; Redis caches it only after the update succeeds.
Cache misses recover from SQL before selecting the human or model path.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from app.db import db
from app.observability.events import (
    EVT_TAKEOVER_CLAIM_CONFLICT,
    EVT_TAKEOVER_CLAIMED,
    EVT_TAKEOVER_CREATED,
    EVT_TAKEOVER_MESSAGE,
)
from app.redis_client import get_redis
from app.services.runtime.tasks import fire_background
from app.services.schedule_domain.schedule import status_label

logger = logging.getLogger(__name__)

STATUS_OVERRIDE_VALUES = frozenset({"idle", "busy", "very_busy", "sleep"})
_PREVIEW_LIMIT = 80


def active_conversation_key(conversation_id: str) -> str:
    return f"takeover:active:{conversation_id}"


def active_user_key(user_id: str) -> str:
    return f"takeover:user:{user_id}"


def status_override_key(conversation_id: str) -> str:
    return f"takeover:status:{conversation_id}"


def _cell(row: Any, *names: str, default=None):
    if row is None:
        return default
    if isinstance(row, dict):
        for name in names:
            if name in row and row[name] is not None:
                return row[name]
        return default
    for name in names:
        if hasattr(row, name):
            value = getattr(row, name)
            if value is not None:
                return value
    return default


def _trim(text: str | None, limit: int) -> str:
    value = " ".join((text or "").split())
    if len(value) <= limit:
        return value
    return f"{value[: max(0, limit - 1)]}…"


async def _active_flag(key: str, column: str, identity: str | None) -> bool:
    if not identity:
        return False
    try:
        redis = await get_redis()
        value = await redis.get(key)
        if value is not None:
            return value != "0"
    except Exception:
        redis = None
    # SQL owns the claim. Cache loss/restart or a partial flag write must not
    # send a claimed user back to the model. Column is an internal allowlist.
    assert column in {"conversation_id", "user_id"}
    try:
        rows = await db.query_raw(
            f"SELECT 1 FROM human_takeovers WHERE {column}=$1 AND status='active' LIMIT 1", identity,
        )
    except Exception as exc:
        logger.warning("takeover lookup unavailable: %s", type(exc).__name__)
        return False
    active = bool(rows)
    if active and redis is not None:
        try:
            await redis.set(key, "1", ex=60, nx=True)
        except Exception:
            pass
    return active


async def is_conversation_taken_over(conversation_id: str | None) -> bool:
    return await _active_flag(active_conversation_key(conversation_id), "conversation_id", conversation_id)


async def is_user_taken_over(user_id: str | None) -> bool:
    return await _active_flag(active_user_key(user_id), "user_id", user_id)


async def display_status_override(conversation_id: str | None) -> str | None:
    """Manual header status. Absent until an admin sets one after claim."""
    if not conversation_id:
        return None
    try:
        redis = await get_redis()
        value = await redis.get(status_override_key(conversation_id))
        if value is None:
            raise LookupError("Status cache miss")
    except Exception:
        try:
            rows = await db.query_raw(
                """
                SELECT status_override
                FROM human_takeovers
                WHERE conversation_id = $1 AND status = 'active'
                LIMIT 1
                """,
                conversation_id,
            )
        except Exception:
            return None
        value = _cell(rows[0], "status_override", "statusOverride") if rows else None
    if value in STATUS_OVERRIDE_VALUES:
        return str(value)
    return None


async def takeover_media_owner(
    actor_user_id: str,
    conversation_id: str,
    *,
    is_admin: bool,
) -> str | None:
    """Conversation owner id when this admin holds the active takeover."""
    if not is_admin or not actor_user_id or not conversation_id:
        return None
    rows = await db.query_raw(
        """
        SELECT user_id AS "userId"
        FROM human_takeovers
        WHERE conversation_id = $1
          AND status = 'active'
          AND owner_admin_id = $2
          AND EXISTS (SELECT 1 FROM users admin WHERE admin.id=$2 AND admin.role='admin'
                      AND admin.status='active' AND admin.archived_at IS NULL)
        LIMIT 1
        """,
        conversation_id,
        actor_user_id,
    )
    if not rows:
        return None
    return str(_cell(rows[0], "userId", "user_id"))


async def open_takeover_case(user_id: str) -> None:
    """Create the pending case for a brand-new account and notify admins.

    Idempotent. Failures stay off the registration path.
    """
    if not user_id:
        return
    try:
        takeover_id = str(uuid.uuid4())
        rows = await db.query_raw(
            """
            WITH new_case AS (
                INSERT INTO human_takeovers (id,user_id,status,unread_count,created_at,updated_at)
                VALUES ($1,$2,'pending',0,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)
                ON CONFLICT (user_id) DO NOTHING RETURNING id
            ), alerts AS (
                INSERT INTO notification_events (user_id,type,title,body,payload,dedupe_key,scheduled_for,updated_at)
                SELECT admin.id,'admin_new_user','新用户注册',
                    LEFT(COALESCE(NULLIF(subject.display_name,''),subject.username,'新用户'),80),
                    jsonb_build_object('type','admin_new_user','route','admin_takeover_review',
                        'takeover_id',new_case.id,'user_id',$2,'title','新用户注册',
                        'body',LEFT(COALESCE(NULLIF(subject.display_name,''),subject.username,'新用户'),80)),
                    new_case.id,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP
                FROM new_case JOIN users subject ON subject.id=$2 CROSS JOIN users admin
                WHERE admin.role='admin' AND admin.status='active' AND admin.archived_at IS NULL
                ON CONFLICT (user_id,type,dedupe_key) DO NOTHING RETURNING id
            ) SELECT id FROM new_case
            """,
            takeover_id,
            user_id,
        )
        if not rows:
            return
        created_id = str(_cell(rows[0], "id") or takeover_id)
        logger.info(
            "takeover case created",
            extra={"event": EVT_TAKEOVER_CREATED, "user_id": user_id, "takeover_id": created_id},
        )
        fire_background(_notify_admins_new_user(created_id, user_id))
    except Exception as exc:
        logger.warning(
            "takeover case create skipped: %s",
            exc,
            extra={"event": EVT_TAKEOVER_CREATED, "user_id": user_id},
        )


async def attach_takeover_agent(
    *,
    user_id: str,
    conversation_id: str,
    agent_id: str,
    workspace_id: str | None,
) -> None:
    """Fill the conversation onto a pending case. Does not claim it."""
    if not user_id or not conversation_id or not agent_id:
        return
    try:
        await db.execute_raw(
            """
            UPDATE human_takeovers
            SET conversation_id = $2,
                agent_id = $3,
                workspace_id = $4,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = $1
              AND status = 'pending'
              AND conversation_id IS NULL
            """,
            user_id,
            conversation_id,
            agent_id,
            workspace_id,
        )
    except Exception as exc:
        logger.warning("takeover agent attach skipped: %s", exc)


async def _notify_admins_new_user(takeover_id: str, user_id: str) -> None:
    user_rows = await db.query_raw(
        """
        SELECT username, display_name AS "displayName"
        FROM users
        WHERE id = $1
        LIMIT 1
        """,
        user_id,
    )
    username = str(_cell(user_rows[0], "username") or "新用户") if user_rows else "新用户"
    display = _cell(user_rows[0], "displayName", "display_name") if user_rows else None
    body = _trim(str(display or username), 80) or "有新用户注册"
    admins = await db.query_raw(
        """
        SELECT id
        FROM users
        WHERE role = 'admin' AND status = 'active' AND archived_at IS NULL
        """
    )
    payload = {
        "type": "admin_new_user",
        "route": "admin_takeover_review",
        "takeover_id": takeover_id,
        "user_id": user_id,
    }
    payload.update(title='新用户注册', body=body)
    from app.services.runtime.ws_manager import manager
    for admin in admins or []:
        admin_id = str(_cell(admin, "id") or "")
        if not admin_id:
            continue
        try:
            # SQL already persisted all recipients atomically with the case.
            await manager.send_to_admin(admin_id, "admin_new_user", payload)
        except Exception as exc:
            logger.warning("new-user inbox delivery failed: %s", type(exc).__name__)
    _kick_notification_dispatch()


def _kick_notification_dispatch() -> None:
    from app.services.notifications.dispatcher import dispatch_due_notifications
    fire_background(dispatch_due_notifications(limit=50))



async def list_takeovers(admin_id: str, *, scope: str) -> list[dict]:
    if scope == "mine":
        rows = await db.query_raw(
            """
            SELECT
                t.id, t.user_id AS "userId", t.conversation_id AS "conversationId",
                t.agent_id AS "agentId", t.status, t.unread_count AS "unreadCount",
                t.last_message_preview AS "lastMessagePreview",
                t.last_user_message_at AS "lastUserMessageAt",
                t.admin_last_read_at AS "adminLastReadAt",
                t.claimed_at AS "claimedAt", t.status_override AS "statusOverride",
                u.username, u.display_name AS "displayName",
                a.name AS "agentName"
            FROM human_takeovers t
            JOIN users u ON u.id = t.user_id
            LEFT JOIN ai_agents a ON a.id = t.agent_id
            WHERE t.status = 'active' AND t.owner_admin_id = $1
            ORDER BY t.last_user_message_at DESC NULLS LAST, t.claimed_at DESC
            LIMIT 100
            """,
            admin_id,
        )
    else:
        rows = await db.query_raw(
            """
            SELECT
                t.id, t.user_id AS "userId", t.conversation_id AS "conversationId",
                t.agent_id AS "agentId", t.status, t.unread_count AS "unreadCount",
                t.created_at AS "createdAt",
                u.username, u.display_name AS "displayName", u.created_at AS "userCreatedAt",
                a.name AS "agentName"
            FROM human_takeovers t
            JOIN users u ON u.id = t.user_id
            LEFT JOIN ai_agents a ON a.id = t.agent_id
            WHERE t.status = 'pending'
              AND NOT EXISTS (
                  SELECT 1 FROM human_takeover_dismissals d
                  WHERE d.takeover_id = t.id AND d.admin_id = $1
              )
            ORDER BY t.created_at DESC
            LIMIT 100
            """,
            admin_id,
        )
    return [_list_item(row) for row in rows or []]


def _list_item(row: Any) -> dict:
    unread = int(_cell(row, "unreadCount", "unread_count") or 0)
    return {
        "id": str(_cell(row, "id")),
        "user_id": str(_cell(row, "userId", "user_id") or ""),
        "conversation_id": _cell(row, "conversationId", "conversation_id"),
        "agent_id": _cell(row, "agentId", "agent_id"),
        "agent_name": _cell(row, "agentName", "agent_name"),
        "username": _cell(row, "username"),
        "display_name": _cell(row, "displayName", "display_name"),
        "status": _cell(row, "status"),
        "unread_count": unread,
        "read_state": "unread" if unread > 0 else "read",
        "last_message_preview": _cell(row, "lastMessagePreview", "last_message_preview"),
        "last_user_message_at": _iso(_cell(row, "lastUserMessageAt", "last_user_message_at")),
        "claimed_at": _iso(_cell(row, "claimedAt", "claimed_at")),
        "created_at": _iso(_cell(row, "createdAt", "created_at") or _cell(row, "userCreatedAt")),
        "status_override": _cell(row, "statusOverride", "status_override"),
        "can_claim": bool(_cell(row, "conversationId", "conversation_id")),
    }


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


async def review_takeover(takeover_id: str) -> dict | None:
    rows = await db.query_raw(
        """
        SELECT
            t.id, t.user_id AS "userId", t.conversation_id AS "conversationId",
            t.agent_id AS "agentId", t.workspace_id AS "workspaceId",
            t.status, t.owner_admin_id AS "ownerAdminId",
            u.username, u.display_name AS "displayName",
            u.signup_source AS "signupSource", u.created_at AS "userCreatedAt",
            a.name AS "agentName", a.gender, a.age, a.city, a.occupation,
            a.background, a.life_overview AS "lifeOverview", a.mbti, a.values,
            a.avatar_url AS "avatarUrl",
            owner.username AS "ownerUsername",
            owner.display_name AS "ownerDisplayName"
        FROM human_takeovers t
        JOIN users u ON u.id = t.user_id
        LEFT JOIN ai_agents a ON a.id = t.agent_id
        LEFT JOIN users owner ON owner.id = t.owner_admin_id
        WHERE t.id = $1
        LIMIT 1
        """,
        takeover_id,
    )
    if not rows:
        return None
    row = rows[0]
    user_id = str(_cell(row, "userId", "user_id"))
    phone = await _masked_phone(user_id)
    mbti = _cell(row, "mbti")
    values = _cell(row, "values")
    return {
        "id": str(_cell(row, "id")),
        "status": _cell(row, "status"),
        "can_claim": _cell(row, "status") == "pending" and bool(_cell(row, "conversationId", "conversation_id")),
        "owner_admin_id": _cell(row, "ownerAdminId", "owner_admin_id"),
        "owner_name": _cell(row, "ownerDisplayName", "owner_display_name")
        or _cell(row, "ownerUsername", "owner_username"),
        "user": {
            "id": user_id,
            "username": _cell(row, "username"),
            "display_name": _cell(row, "displayName", "display_name"),
            "signup_source": _cell(row, "signupSource", "signup_source"),
            "created_at": _iso(_cell(row, "userCreatedAt", "user_created_at")),
            "phone_masked": phone,
        },
        "agent": None if not _cell(row, "agentId", "agent_id") else {
            "id": _cell(row, "agentId", "agent_id"),
            "name": _cell(row, "agentName", "agent_name"),
            "gender": _cell(row, "gender"),
            "age": _cell(row, "age"),
            "city": _cell(row, "city"),
            "occupation": _cell(row, "occupation"),
            "background": _cell(row, "background"),
            "life_overview": _cell(row, "lifeOverview", "life_overview"),
            "mbti": mbti,
            "values": values,
            "avatar_url": _cell(row, "avatarUrl", "avatar_url"),
            "conversation_id": _cell(row, "conversationId", "conversation_id"),
            "workspace_id": _cell(row, "workspaceId", "workspace_id"),
        },
    }


async def _masked_phone(user_id: str) -> str | None:
    rows = await db.query_raw(
        """
        SELECT provider_account_id AS "phone"
        FROM auth_identities
        WHERE user_id = $1 AND provider = 'phone'
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        user_id,
    )
    phone = str(_cell(rows[0], "phone") or "") if rows else ""
    digits = "".join(ch for ch in phone if ch.isdigit())
    if len(digits) < 7:
        return None
    return f"{digits[:3]}****{digits[-4:]}"


async def dismiss_takeover(takeover_id: str, admin_id: str) -> bool:
    rows = await db.query_raw(
        """
        SELECT status FROM human_takeovers WHERE id = $1 LIMIT 1
        """,
        takeover_id,
    )
    if not rows or _cell(rows[0], "status") != "pending":
        return False
    await db.execute_raw(
        """
        INSERT INTO human_takeover_dismissals (id, takeover_id, admin_id, created_at)
        VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
        ON CONFLICT (takeover_id, admin_id) DO NOTHING
        """,
        str(uuid.uuid4()),
        takeover_id,
        admin_id,
    )
    return True


class TakeoverClaimError(Exception):
    def __init__(self, code: str, *, owner_name: str | None = None):
        super().__init__(code)
        self.code = code
        self.owner_name = owner_name


async def claim_takeover(takeover_id: str, admin_id: str) -> dict:
    rows = await db.query_raw(
        """
        UPDATE human_takeovers
        SET status = 'active',
            owner_admin_id = $1,
            claimed_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $2
          AND status = 'pending'
          AND conversation_id IS NOT NULL
          AND EXISTS (SELECT 1 FROM users admin WHERE admin.id=$1
                      AND admin.role='admin' AND admin.status='active' AND admin.archived_at IS NULL)
          AND EXISTS (
              SELECT 1 FROM conversations c
              JOIN ai_agents a ON a.id=c.agent_id AND a.user_id=c.user_id
              JOIN chat_workspaces w ON w.id=c.workspace_id AND w.agent_id=a.id AND w.user_id=c.user_id
              JOIN users u ON u.id=c.user_id
              WHERE c.id=human_takeovers.conversation_id AND c.user_id=human_takeovers.user_id
                AND c.agent_id=human_takeovers.agent_id AND c.workspace_id=human_takeovers.workspace_id
                AND NOT c.is_deleted AND c.archived_at IS NULL AND a.status='active'
                AND a.archived_at IS NULL AND w.status='active' AND w.archived_at IS NULL
                AND u.status='active' AND u.archived_at IS NULL
          )
        RETURNING
            id, user_id AS "userId", conversation_id AS "conversationId",
            agent_id AS "agentId", workspace_id AS "workspaceId"
        """,
        admin_id,
        takeover_id,
    )
    if not rows:
        conflict = await db.query_raw(
            """
            SELECT t.status, t.conversation_id AS "conversationId",
                   owner.display_name AS "ownerDisplayName",
                   owner.username AS "ownerUsername"
            FROM human_takeovers t
            LEFT JOIN users owner ON owner.id = t.owner_admin_id
            WHERE t.id = $1
            LIMIT 1
            """,
            takeover_id,
        )
        if not conflict:
            raise TakeoverClaimError("not_found")
        status = _cell(conflict[0], "status")
        if status == "active":
            owner_name = (
                _cell(conflict[0], "ownerDisplayName", "owner_display_name")
                or _cell(conflict[0], "ownerUsername", "owner_username")
                or "其他管理员"
            )
            logger.info(
                "takeover claim conflict",
                extra={
                    "event": EVT_TAKEOVER_CLAIM_CONFLICT,
                    "takeover_id": takeover_id,
                    "admin_id": admin_id,
                },
            )
            raise TakeoverClaimError("already_taken_over", owner_name=str(owner_name))
        if not _cell(conflict[0], "conversationId", "conversation_id"):
            raise TakeoverClaimError("agent_not_ready")
        raise TakeoverClaimError("not_pending")
    row = rows[0]
    conversation_id = str(_cell(row, "conversationId", "conversation_id"))
    user_id = str(_cell(row, "userId", "user_id"))
    try:
        await _write_active_flags(conversation_id, user_id)
    except Exception as exc:
        # The committed SQL row is authoritative, including on Redis misses.
        logger.warning("takeover cache publication failed: %s", type(exc).__name__)
    logger.info(
        "takeover claimed",
        extra={
            "event": EVT_TAKEOVER_CLAIMED,
            "takeover_id": takeover_id,
            "admin_id": admin_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
        },
    )
    detail = await review_takeover(takeover_id)
    return detail or {"id": takeover_id, "status": "active"}


async def _write_active_flags(conversation_id: str, user_id: str) -> None:
    redis = await get_redis()
    async with redis.pipeline(transaction=True) as pipeline:
        pipeline.set(active_conversation_key(conversation_id), "1")
        pipeline.set(active_user_key(user_id), "1")
        await pipeline.execute()


async def set_status_override(takeover_id: str, admin_id: str, status: str) -> dict:
    if status not in STATUS_OVERRIDE_VALUES:
        raise ValueError("invalid_status")
    rows = await db.query_raw(
        """
        UPDATE human_takeovers
        SET status_override = $3, updated_at = CURRENT_TIMESTAMP
        WHERE id = $1 AND owner_admin_id = $2 AND status = 'active'
        RETURNING conversation_id AS "conversationId", status_override AS "statusOverride"
        """,
        takeover_id,
        admin_id,
        status,
    )
    if not rows:
        raise TakeoverClaimError("not_owner")
    conversation_id = str(_cell(rows[0], "conversationId", "conversation_id"))
    try:
        redis = await get_redis()
        await redis.set(status_override_key(conversation_id), status)
    except Exception as exc:
        logger.warning("takeover status flag write failed: %s", exc)
    label = status_label(status)
    from app.services.runtime.ws_manager import manager

    await manager.send_event(
        conversation_id,
        "ai_status",
        {"ai_status": status, "ai_status_label": label},
    )
    return {
        "conversation_id": conversation_id,
        "ai_status": status,
        "ai_status_label": label,
    }


async def takeover_unread_summary(admin_id: str) -> dict:
    rows = await db.query_raw("""
        SELECT COALESCE(sum(unread_count),0)::bigint AS unread_count,
          count(*) FILTER (WHERE unread_count>0)::int AS unread_chats,
          CURRENT_TIMESTAMP AS sampled_at
        FROM human_takeovers WHERE owner_admin_id=$1 AND status='active'
    """, admin_id)
    row = rows[0]
    return {"unread_count": int(row['unread_count']), "unread_chats": row['unread_chats'],
            "sampled_at": str(row['sampled_at'])}


async def mark_takeover_read(takeover_id: str, admin_id: str, message_id: str) -> bool:
    # A receipt only acknowledges the delivered prefix, never messages that
    # arrived after that snapshot. Monotonic positions serialize with ingress.
    rows = await db.query_raw("""
        UPDATE human_takeovers t
        SET admin_read_seq=GREATEST(t.admin_read_seq,(m.metadata->>'takeover_sequence')::bigint),
            unread_count=t.user_message_seq-GREATEST(t.admin_read_seq,(m.metadata->>'takeover_sequence')::bigint),
            admin_last_read_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP
        FROM messages m
        WHERE t.id=$1 AND t.owner_admin_id=$2 AND t.status='active'
          AND m.id=$3 AND m.conversation_id=t.conversation_id AND m.role='user'
          AND m.metadata->>'human_takeover'='true'
          AND (m.metadata->>'takeover_sequence')::bigint BETWEEN 1 AND t.user_message_seq
        RETURNING t.id
    """, takeover_id, admin_id, message_id)
    return bool(rows)


async def record_takeover_user_message(
    *,
    conversation_id: str,
    user_id: str,
    text: str,
    metadata: dict | None,
    paid_confirmed: bool = False,
) -> dict:
    """Persist one post-claim user message and notify the owning admin.

    Invalid or no longer active scopes fail before committing a message.
    """
    from app.services.admin_takeover.ingress import persist_user_message

    saved = await persist_user_message(
        conversation_id=conversation_id, user_id=user_id, text=text,
        metadata=metadata or {}, paid_confirmed=paid_confirmed, database=db,
    )
    if not saved["created"]:
        return {"message_id": saved["message_id"], "duplicate": True}
    row = saved["row"]
    message_id = saved["message_id"]
    preview = _trim(text, _PREVIEW_LIMIT) or "［新消息］"
    owner_id = str(_cell(row, "ownerAdminId", "owner_admin_id") or "")
    takeover_id = str(_cell(row, "id"))
    payload = {
        "takeover_id": takeover_id,
        "conversation_id": conversation_id,
        "user_id": user_id,
        "message_id": message_id,
        "text": text,
        "preview": preview,
        "user_name": _cell(row, "user_name") or "用户",
        "metadata": saved["metadata"],
        "created_at": saved["created_at"],
        "role": "user",
        "unread_count": int(_cell(row, "unreadCount", "unread_count") or 0),
    }
    logger.info(
        "takeover user message routed",
        extra={
            "event": EVT_TAKEOVER_MESSAGE,
            "takeover_id": takeover_id,
            "conversation_id": conversation_id,
            "message_id": message_id,
            "text_chars": len(text),
        },
    )
    if owner_id:
        from app.services.runtime.ws_manager import manager

        try:
            await manager.send_to_admin(owner_id, "takeover_user_message", payload)

        except Exception as exc:
            logger.warning("takeover notification failed after commit: %s", type(exc).__name__)
    _kick_notification_dispatch()
    return {"message_id": message_id, "owner_admin_id": owner_id, "takeover_id": takeover_id}


async def _push_owner_message(
    admin_id: str,
    takeover_id: str,
    conversation_id: str,
    message_id: str,
    preview: str,
    *, user_name: str = "用户", database=None, dispatch_now: bool = True,
) -> None:
    from app.services.notifications.service import enqueue_notification

    await enqueue_notification(
        user_id=admin_id,
        type="admin_takeover_message",
        title=_trim(user_name, 80) or "用户",
        body=preview or "你收到了一条新消息",
        payload={
            "type": "admin_takeover_message",
            "route": "admin_takeover_chat",
            "takeover_id": takeover_id,
            "conversation_id": conversation_id,
            "message_id": message_id,
            "title": _trim(user_name,80) or "用户",
            "body": preview or "你收到了一条新消息",
        },
        database=database,
        dispatch_now=dispatch_now,
        conversation_id=conversation_id,
        message_id=message_id,
        dedupe_key=message_id,
    )


async def send_admin_message(
    *,
    takeover_id: str,
    admin_id: str,
    text: str,
    component_card: dict | None = None,
    attachments: list[dict] | None = None,
    client_id: str | None = None,
) -> dict:
    from app.services.admin_takeover.ingress import persist_admin_message

    saved = await persist_admin_message(takeover_id=takeover_id, admin_id=admin_id, text=text,
        component_card=component_card, attachments=attachments, client_id=client_id, database=db)
    scope = saved["scope"]
    conversation_id, user_id = scope.conversation_id, scope.owner_user_id
    agent_id, workspace_id = scope.agent_id, scope.workspace_id
    message_id = saved["message_id"]
    if not saved["created"]:
        return {"message_id": message_id, "conversation_id": conversation_id, "duplicate": True}
    attachments = saved["metadata"].get("attachments", [])
    reply = {
        "text": text,
        "index": 0,
        "assistant_message_id": message_id,
        "message_id": message_id,
    }
    if component_card:
        reply["component_card"] = component_card
    if attachments:
        reply["attachments"] = attachments
    from app.services.notifications.service import notify_agent_message_created
    from app.services.runtime.ws_manager import manager

    await manager.send_event(conversation_id, "reply", reply)
    await manager.send_event(conversation_id, "done", {"message_id": message_id})
    await manager.send_to_admin(
        admin_id,
        "takeover_admin_message",
        {
            "takeover_id": takeover_id,
            "conversation_id": conversation_id,
            "message_id": message_id,
            "text": text,
            "role": "assistant",
            "component_card": component_card,
            "attachments": attachments or [],
        },
    )
    fire_background(
        notify_agent_message_created(
            conversation_id=conversation_id,
            message_id=message_id,
            text=text or "你收到了一条新消息",
            metadata={"human_takeover": True},
            user_id=user_id,
            agent_id=str(agent_id) if agent_id else None,
            workspace_id=str(workspace_id) if workspace_id else None,
        )
    )
    logger.info(
        "takeover admin reply routed",
        extra={
            "event": EVT_TAKEOVER_MESSAGE,
            "takeover_id": takeover_id,
            "conversation_id": conversation_id,
            "message_id": message_id,
            "text_chars": len(text),
            "direction": "admin_to_user",
        },
    )
    return {"message_id": message_id, "conversation_id": conversation_id}
