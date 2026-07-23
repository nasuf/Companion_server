"""Shared helpers for the native in-app game sessions.

These utilities used to live alongside the (now removed) SUD provider; they are
provider-agnostic session plumbing reused by ``games.native``: player DTO
construction, owned-context resolution, idempotent event append, chat message
projection, and row → session mapping.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any
from weakref import WeakValueDictionary

from app.db import db
from app.models.game import GamePlayerInfo, GameSessionRow, NativeSessionResponse

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = {"settled", "aborted"}
_GAME_STATUS_LOCKS: WeakValueDictionary[tuple[str, str, str], asyncio.Lock] = (
    WeakValueDictionary()
)
_GAME_REPLY_LOCKS: WeakValueDictionary[tuple[str, str, str], asyncio.Lock] = (
    WeakValueDictionary()
)

GameSessionResponse = GameSessionRow | NativeSessionResponse

_NATIVE_TITLES = {
    "go": "围棋",
    "reversi": "黑白棋",
    "gomoku": "五子棋",
    "xiangqi": "中国象棋",
    "chess": "国际象棋",
    "chinese_checkers": "跳棋",
    "match3": "消消乐",
    "minesweeper": "协作扫雷",
    "number_merge": "数字合并",
    "tetris_duel": "双人方块竞速",
}


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _loads(value: Any, fallback: Any = None) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return fallback
    return fallback


def ai_level_for_difficulty(difficulty: str) -> int:
    if difficulty == "hard":
        return 3
    if difficulty == "normal":
        return 2
    return 1


def _avatar_for_initial(name: str, hue: str) -> str:
    initial = (name or "C").strip()[:1] or "C"
    return (
        "https://api.dicebear.com/9.x/initials/svg?"
        f"seed={initial}&backgroundColor={hue}&fontFamily=Arial"
    )


def _gender_for_ai(raw: str | None) -> str:
    value = (raw or "").strip().lower()
    if value in {"male", "female"}:
        return value
    return ""


async def build_user_player(user_id: str) -> GamePlayerInfo:
    user = await db.user.find_unique(where={"id": user_id})
    username = getattr(user, "username", None) or "玩家"
    return GamePlayerInfo(
        uid=user_id,
        nick_name=username,
        avatar_url=_avatar_for_initial(username, "38bdf8"),
        gender="",
        is_ai=0,
        ai_level=0,
    )


async def build_ai_player(
    agent_id: str,
    difficulty: str,
    agent: Any | None = None,
) -> GamePlayerInfo:
    if agent is None:
        agent = await db.aiagent.find_unique(where={"id": agent_id})
    name = getattr(agent, "name", None) or "Companion"
    return GamePlayerInfo(
        uid=f"agent:{agent_id}",
        nick_name=name,
        avatar_url=getattr(agent, "avatarUrl", None)
        or _avatar_for_initial(name, "f97316"),
        gender=_gender_for_ai(getattr(agent, "gender", None)),
        is_ai=1,
        ai_level=ai_level_for_difficulty(difficulty),
    )


async def _resolve_owned_context(
    *,
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
    conversation_id: str | None,
) -> tuple[str | None, str | None]:
    resolved_workspace_id = workspace_id
    if conversation_id:
        rows = await db.query_raw(
            """
            SELECT
                c.id AS conversation_id,
                c.user_id AS conversation_user_id,
                c.agent_id AS conversation_agent_id,
                c.workspace_id AS conversation_workspace_id,
                w.id AS workspace_id,
                w.user_id AS workspace_user_id,
                w.agent_id AS workspace_agent_id
            FROM conversations c
            LEFT JOIN chat_workspaces w
              ON w.id = COALESCE(c.workspace_id, $2)
            WHERE c.id = $1 AND c.is_deleted = FALSE
            LIMIT 1
            """,
            conversation_id,
            resolved_workspace_id,
        )
        if not rows:
            raise ValueError("context_not_found")
        conversation = rows[0]
        if (
            conversation.get("conversation_user_id") != user_id
            or conversation.get("conversation_agent_id") != agent_id
        ):
            raise ValueError("context_not_found")
        conversation_workspace_id = conversation.get("conversation_workspace_id")
        if resolved_workspace_id and conversation_workspace_id != resolved_workspace_id:
            raise ValueError("context_not_found")
        resolved_workspace_id = conversation_workspace_id or resolved_workspace_id
        if resolved_workspace_id:
            if conversation.get("workspace_id") != resolved_workspace_id:
                raise ValueError("context_not_found")
            workspace_agent_id = conversation.get("workspace_agent_id")
            if conversation.get("workspace_user_id") != user_id:
                raise ValueError("context_not_found")
            if workspace_agent_id and workspace_agent_id != agent_id:
                raise ValueError("context_not_found")
    elif resolved_workspace_id:
        rows = await db.query_raw(
            """
            SELECT id, user_id, agent_id
            FROM chat_workspaces
            WHERE id = $1
            LIMIT 1
            """,
            resolved_workspace_id,
        )
        if not rows:
            raise ValueError("context_not_found")
        workspace = rows[0]
        workspace_agent_id = workspace.get("agent_id")
        if workspace.get("user_id") != user_id:
            raise ValueError("context_not_found")
        if workspace_agent_id and workspace_agent_id != agent_id:
            raise ValueError("context_not_found")

    return resolved_workspace_id, conversation_id


async def _append_event_idempotent(
    *,
    session_id: str,
    event_type: str,
    state: str | None,
    payload: dict[str, Any],
    source: str,
    companion_reply: str | None,
    client_event_id: str | None,
    database: Any | None = None,
) -> tuple[str, bool, str | None]:
    executor = database or db
    event_id = str(uuid.uuid4())
    rows = await executor.query_raw(
        """
        INSERT INTO game_events (
            id, session_id, client_event_id, event_type, state, source,
            payload, companion_reply
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8)
        ON CONFLICT (session_id, client_event_id)
            WHERE client_event_id IS NOT NULL
            DO NOTHING
        RETURNING id, companion_reply
        """,
        event_id,
        session_id,
        client_event_id,
        event_type,
        state,
        source,
        _json(payload),
        companion_reply,
    )
    if rows:
        return str(rows[0]["id"]), True, rows[0].get("companion_reply")
    existing = await executor.query_raw(
        """
        SELECT id, companion_reply
        FROM game_events
        WHERE session_id = $1 AND client_event_id = $2
        LIMIT 1
        """,
        session_id,
        client_event_id,
    )
    if not existing:
        raise RuntimeError("idempotent game event insert returned no row")
    return (
        str(existing[0]["id"]),
        False,
        existing[0].get("companion_reply"),
    )


async def _write_game_message(
    *,
    conversation_id: str,
    role: str,
    content: str,
    metadata: dict[str, Any],
) -> tuple[str, bool]:
    """Insert one logical game message across all API workers."""

    message_id = str(uuid.uuid4())
    dedupe_keys = (
        ("kind", "session_id", "game_status")
        if metadata.get("kind") == "game_status"
        else ("kind", "session_id", "event_type")
    )
    dedupe = {key: metadata[key] for key in dedupe_keys if key in metadata}
    lock_key = f"game-message:{conversation_id}:{_json(dedupe)}"
    async with db.tx() as tx:
        # pg_advisory_xact_lock returns void; select it from FROM so query_raw
        # gets a real (int) column instead of failing to deserialize `void`.
        await tx.query_raw(
            "SELECT 1 AS locked FROM pg_advisory_xact_lock(hashtextextended($1, 0))",
            lock_key,
        )
        existing = await tx.query_raw(
            """
            SELECT id
            FROM messages
            WHERE conversation_id = $1
              AND metadata @> $2::jsonb
            LIMIT 1
            """,
            conversation_id,
            _json(dedupe),
        )
        if existing:
            return str(existing[0]["id"]), False
        rows = await tx.query_raw(
            """
            INSERT INTO messages (id, conversation_id, role, content, metadata)
            VALUES ($1, $2, $3, $4, $5::jsonb)
            RETURNING id
            """,
            message_id,
            conversation_id,
            role,
            content,
            _json(metadata),
        )
    if not rows:
        raise RuntimeError("game chat message insert returned no row")
    return str(rows[0]["id"]), True


def _status_transition(
    previous: GameSessionResponse,
    updated: GameSessionResponse,
    event_type: str,
    state: str | None,
) -> str | None:
    if updated.status == "playing" and previous.status != "playing":
        return "started"
    if (
        updated.status in _TERMINAL_STATUSES
        and previous.status not in _TERMINAL_STATUSES
    ):
        return "ended"
    return None


def _game_title(session: GameSessionResponse, payload: dict[str, Any]) -> str:
    title = str(payload.get("game_title") or payload.get("gameName") or "").strip()
    if title:
        return title
    return _NATIVE_TITLES.get(getattr(session, "game_key", None), "游戏")


def _ended_reason(
    session: GameSessionResponse,
    event_type: str,
    state: str | None,
    payload: dict[str, Any],
) -> str:
    if session.status == "aborted":
        return str(payload.get("reason") or payload.get("exit_reason") or "aborted")
    result = session.result or {}
    raw_reason = (
        result.get("game_over_reason")
        or payload.get("gameOverReason")
        or payload.get("game_over_reason")
    )
    return str(raw_reason or "settled")


def _should_persist_reply_to_chat(event_type: str, state: str | None) -> bool:
    return event_type in {"game_finished", "game_aborted"}


async def _persist_game_status_to_chat_if_needed(
    previous: GameSessionResponse,
    updated: GameSessionResponse,
    event_type: str,
    state: str | None,
    payload: dict[str, Any],
) -> None:
    if not updated.conversation_id:
        return
    status = _status_transition(previous, updated, event_type, state)
    if status is None:
        return
    lock_key = (updated.conversation_id, updated.id, status)
    lock = _GAME_STATUS_LOCKS.setdefault(lock_key, asyncio.Lock())
    async with lock:
        game_title = _game_title(updated, payload)
        actor_name = updated.ai_player.nick_name or "AI"
        text = f"{actor_name} 和你已{'退出' if status == 'ended' else '进入'}游戏《{game_title}》"
        metadata: dict[str, Any] = {
            "kind": "game_status",
            "game_status": status,
            "game_title": game_title,
            "game_status_actor": "both",
            "game_status_actor_name": actor_name,
            "session_id": updated.id,
            "event_type": event_type,
            "state": state,
        }
        if status == "ended":
            metadata["game_ended_reason"] = _ended_reason(
                updated, event_type, state, payload
            )
        message_id, inserted = await _write_game_message(
            conversation_id=updated.conversation_id,
            role="assistant",
            content=text,
            metadata=metadata,
        )
        if not inserted:
            return
        try:
            from app.services.runtime.ws_manager import manager

            event_payload: dict[str, Any] = {
                "text": text,
                "status": status,
                "game_title": game_title,
                "session_id": updated.id,
                "message_id": message_id or "",
                "actor": "both",
                "actor_name": actor_name,
                "reason": metadata.get("game_ended_reason", ""),
            }
            await manager.send_event(
                updated.conversation_id,
                "game_status",
                event_payload,
            )
        except Exception as exc:
            logger.debug("failed to emit game status websocket event: %r", exc)


async def _persist_reply_to_chat_if_needed(
    session: GameSessionResponse,
    event_type: str,
    state: str | None,
    reply: str | None,
) -> None:
    if not reply or not session.conversation_id:
        return
    if not _should_persist_reply_to_chat(event_type, state):
        return
    lock_key = (session.conversation_id, session.id, event_type)
    lock = _GAME_REPLY_LOCKS.setdefault(lock_key, asyncio.Lock())
    async with lock:
        message_id, inserted = await _write_game_message(
            conversation_id=session.conversation_id,
            role="assistant",
            content=reply,
            metadata={
                "kind": "game",
                "session_id": session.id,
                "event_type": event_type,
                "state": state,
            },
        )
        if not inserted:
            return
    try:
        from app.services.runtime.ws_manager import manager

        await manager.send_event(
            session.conversation_id,
            "proactive",
            {
                "text": reply,
                "assistant_message_id": message_id or "",
                "trigger_type": "game",
                "session_id": session.id,
                "event_type": event_type,
                "state": state,
            },
        )
    except Exception as exc:
        logger.debug("failed to emit game reply websocket event: %r", exc)


def _row_to_session(row: dict[str, Any]) -> GameSessionRow:
    user_player = GamePlayerInfo.model_validate(_loads(row.get("user_player"), {}))
    ai_player = GamePlayerInfo.model_validate(_loads(row.get("ai_player"), {}))
    return GameSessionRow(
        id=str(row["id"]),
        provider=str(row.get("provider") or "native"),
        game_key=row.get("game_key"),
        status=str(row.get("status") or "created"),
        user_id=str(row.get("user_id") or ""),
        agent_id=str(row.get("agent_id") or ""),
        workspace_id=row.get("workspace_id"),
        conversation_id=row.get("conversation_id"),
        room_id=str(row.get("room_id") or ""),
        play_mode=str(row.get("play_mode") or "versus"),  # type: ignore[arg-type]
        difficulty=str(row.get("difficulty") or "normal"),  # type: ignore[arg-type]
        ai_level=int(row.get("ai_level") or 0),
        user_player=user_player,
        ai_player=ai_player,
        companion_reply=row.get("companion_reply"),
        result=_loads(row.get("result"), None),
        duration_seconds=row.get("duration_seconds"),
        started_at=_iso(row.get("started_at")),
        ended_at=_iso(row.get("ended_at")),
        created_at=_iso(row.get("created_at")),
    )
