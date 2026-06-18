from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from prisma import Json

from app.config import settings
from app.db import db
from app.models.game import SudPlayerInfo, SudSessionResponse

logger = logging.getLogger(__name__)

CODE_TTL_SECONDS = 30 * 60
TOKEN_TTL_SECONDS = 2 * 60 * 60
DEFAULT_DEMO_MG_ID = "1461227817776713818"
GOMOKU_MG_ID = "1676069429630722049"
MONSTER_CRUSH_MG_ID = "1664525565526667266"
_TERMINAL_STATUSES = {"settled", "aborted"}
_GAME_STATUS_LOCKS: dict[tuple[str, str, str], asyncio.Lock] = {}


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


def _signing_secret() -> str:
    return (
        settings.sud_app_secret.strip()
        or settings.jwt_secret.strip()
        or "companion-sud-demo-secret"
    )


def sdk_enabled() -> bool:
    return bool(
        settings.sud_app_id.strip()
        and settings.sud_app_key.strip()
        and settings.sud_app_secret.strip()
        and settings.sud_default_mg_id.strip()
    )


def missing_config() -> list[str]:
    missing: list[str] = []
    for key in ("sud_app_id", "sud_app_key", "sud_app_secret", "sud_default_mg_id"):
        if not getattr(settings, key).strip():
            missing.append(key.upper())
    return missing


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


def _gender_for_sud(raw: str | None) -> str:
    value = (raw or "").strip().lower()
    if value in {"male", "female"}:
        return value
    return ""


async def build_user_player(user_id: str) -> SudPlayerInfo:
    user = await db.user.find_unique(where={"id": user_id})
    username = getattr(user, "username", None) or "玩家"
    return SudPlayerInfo(
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
) -> SudPlayerInfo:
    if agent is None:
        agent = await db.aiagent.find_unique(where={"id": agent_id})
    name = getattr(agent, "name", None) or "Companion"
    return SudPlayerInfo(
        uid=f"agent:{agent_id}",
        nick_name=name,
        avatar_url=getattr(agent, "avatarUrl", None)
        or _avatar_for_initial(name, "f97316"),
        gender=_gender_for_sud(getattr(agent, "gender", None)),
        is_ai=1,
        ai_level=ai_level_for_difficulty(difficulty),
    )


def make_code(*, user_id: str, session_id: str, room_id: str) -> tuple[str, datetime]:
    expires_at = _now() + timedelta(seconds=CODE_TTL_SECONDS)
    payload = {
        "uid": user_id,
        "session_id": session_id,
        "room_id": room_id,
        "app_id": settings.sud_app_id.strip() or "companion-demo",
        "exp": int(expires_at.timestamp()),
    }
    return jwt.encode(payload, _signing_secret(), algorithm="HS256"), expires_at


def make_ss_token(
    *,
    uid: str,
    session_id: str | None,
    room_id: str | None,
) -> tuple[str, datetime]:
    expires_at = _now() + timedelta(seconds=TOKEN_TTL_SECONDS)
    payload = {
        "uid": uid,
        "session_id": session_id,
        "room_id": room_id,
        "app_id": settings.sud_app_id.strip() or "companion-demo",
        "exp": int(expires_at.timestamp()),
    }
    return jwt.encode(payload, _signing_secret(), algorithm="HS256"), expires_at


def decode_token(token: str) -> dict[str, Any]:
    return jwt.decode(token, _signing_secret(), algorithms=["HS256"])


async def create_session(
    *,
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
    conversation_id: str | None,
    mg_id: str | None,
    room_id: str | None,
    play_mode: str,
    difficulty: str,
) -> SudSessionResponse:
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent or getattr(agent, "userId", None) != user_id:
        raise ValueError("agent_not_found")
    workspace_id, conversation_id = await _resolve_owned_context(
        user_id=user_id,
        agent_id=agent_id,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
    )

    session_id = str(uuid.uuid4())
    resolved_room_id = room_id or f"cmp-{session_id[:8]}"
    resolved_mg_id = mg_id or settings.sud_default_mg_id.strip() or DEFAULT_DEMO_MG_ID
    code, code_expires_at = make_code(
        user_id=user_id,
        session_id=session_id,
        room_id=resolved_room_id,
    )
    user_player = await build_user_player(user_id)
    ai_player = await build_ai_player(agent_id, difficulty, agent=agent)
    companion_reply = _intro_reply(ai_player.nick_name, play_mode, difficulty)

    await db.execute_raw(
        """
        INSERT INTO game_sessions (
            id, provider, status, user_id, agent_id, workspace_id, conversation_id,
            mg_id, room_id, play_mode, difficulty, ai_level, sdk_enabled,
            sud_code, sud_code_expires_at, user_player, ai_player, companion_reply
        )
        VALUES (
            $1, 'sud', 'created', $2, $3, $4, $5,
            $6, $7, $8, $9, $10, $11,
            $12, $13::timestamptz, $14::jsonb, $15::jsonb, $16
        )
        """,
        session_id,
        user_id,
        agent_id,
        workspace_id,
        conversation_id,
        resolved_mg_id,
        resolved_room_id,
        play_mode,
        difficulty,
        ai_player.ai_level,
        sdk_enabled(),
        code,
        code_expires_at,
        user_player.model_dump_json(),
        ai_player.model_dump_json(),
        companion_reply,
    )
    await _append_event(
        session_id=session_id,
        event_type="session_created",
        state=None,
        payload={
            "play_mode": play_mode,
            "difficulty": difficulty,
            "ai_level": ai_player.ai_level,
            "mg_id": resolved_mg_id,
        },
        source="server",
        companion_reply=companion_reply,
    )
    return await get_session(session_id, user_id=user_id)


async def refresh_code(session_id: str, *, user_id: str) -> SudSessionResponse:
    session = await get_session(session_id, user_id=user_id)
    code, expires_at = make_code(
        user_id=session.user_id,
        session_id=session.id,
        room_id=session.room_id,
    )
    await db.execute_raw(
        """
        UPDATE game_sessions
        SET sud_code = $2, sud_code_expires_at = $3::timestamptz, updated_at = CURRENT_TIMESTAMP
        WHERE id = $1
        """,
        session_id,
        code,
        expires_at,
    )
    return await get_session(session_id, user_id=user_id)


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
            SELECT id, user_id, agent_id, workspace_id
            FROM conversations
            WHERE id = $1 AND is_deleted = FALSE
            LIMIT 1
            """,
            conversation_id,
        )
        if not rows:
            raise ValueError("context_not_found")
        conversation = rows[0]
        if (
            conversation.get("user_id") != user_id
            or conversation.get("agent_id") != agent_id
        ):
            raise ValueError("context_not_found")
        conversation_workspace_id = conversation.get("workspace_id")
        if resolved_workspace_id and conversation_workspace_id != resolved_workspace_id:
            raise ValueError("context_not_found")
        resolved_workspace_id = conversation_workspace_id or resolved_workspace_id

    if resolved_workspace_id:
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


async def list_sessions(user_id: str, limit: int = 50) -> list[SudSessionResponse]:
    rows = await db.query_raw(
        """
        SELECT *
        FROM game_sessions
        WHERE user_id = $1
        ORDER BY created_at DESC
        LIMIT $2
        """,
        user_id,
        limit,
    )
    return [_row_to_session(row) for row in rows]


async def get_session(session_id: str, *, user_id: str | None = None) -> SudSessionResponse:
    if user_id:
        rows = await db.query_raw(
            "SELECT * FROM game_sessions WHERE id = $1 AND user_id = $2 LIMIT 1",
            session_id,
            user_id,
        )
    else:
        rows = await db.query_raw(
            "SELECT * FROM game_sessions WHERE id = $1 LIMIT 1",
            session_id,
        )
    if not rows:
        raise ValueError("session_not_found")
    return _row_to_session(rows[0])


async def handle_client_event(
    *,
    session_id: str,
    user_id: str,
    event_type: str,
    state: str | None,
    payload: dict[str, Any],
    source: str,
) -> tuple[SudSessionResponse, str | None, str]:
    session = await get_session(session_id, user_id=user_id)
    reply = _reply_for_event(session, event_type, state, payload)
    await _update_session_from_event(session, event_type, state, payload, reply)
    event_id = await _append_event(
        session_id=session_id,
        event_type=event_type,
        state=state,
        payload=payload,
        source=source,
        companion_reply=reply,
    )
    updated = await get_session(session_id, user_id=user_id)
    await _persist_game_status_to_chat_if_needed(session, updated, event_type, state, payload)
    await _persist_reply_to_chat_if_needed(updated, event_type, state, reply)
    return updated, reply, event_id


async def handle_sud_report(
    report_type: str,
    report_msg: dict[str, Any],
) -> SudSessionResponse | None:
    room_id = str(report_msg.get("room_id") or "")
    if not room_id:
        return None
    rows = await db.query_raw(
        "SELECT * FROM game_sessions WHERE room_id = $1 ORDER BY created_at DESC LIMIT 1",
        room_id,
    )
    if not rows:
        logger.info("SUD report for unknown room_id=%s", room_id)
        return None
    session = _row_to_session(rows[0])
    event_type = f"sud_{report_type}"
    reply = _reply_for_event(session, event_type, None, report_msg)
    await _update_session_from_event(session, event_type, None, report_msg, reply)
    await _append_event(
        session_id=session.id,
        event_type=event_type,
        state=None,
        payload=report_msg,
        source="sud_callback",
        companion_reply=reply,
    )
    updated = await get_session(session.id)
    await _persist_game_status_to_chat_if_needed(session, updated, event_type, None, report_msg)
    await _persist_reply_to_chat_if_needed(updated, event_type, None, reply)
    return updated


async def handle_sud_notify(
    notify_event: str,
    data: dict[str, Any],
) -> SudSessionResponse | None:
    room_id = str(data.get("room_id") or data.get("roomId") or "")
    if not room_id:
        return None
    rows = await db.query_raw(
        "SELECT * FROM game_sessions WHERE room_id = $1 ORDER BY created_at DESC LIMIT 1",
        room_id,
    )
    if not rows:
        logger.info("SUD notify for unknown room_id=%s event=%s", room_id, notify_event)
        return None
    session = _row_to_session(rows[0])
    event_type = _event_type_for_notify(notify_event)
    state = str(data.get("event") or data.get("state") or "") or None
    reply = _reply_for_event(session, event_type, state, data)
    await _update_session_from_event(session, event_type, state, data, reply)
    await _append_event(
        session_id=session.id,
        event_type=event_type,
        state=state,
        payload={"notify_event": notify_event, **data},
        source="sud_callback",
        companion_reply=reply,
    )
    updated = await get_session(session.id)
    await _persist_game_status_to_chat_if_needed(session, updated, event_type, state, data)
    await _persist_reply_to_chat_if_needed(updated, event_type, state, reply)
    return updated


def _event_type_for_notify(notify_event: str) -> str:
    if notify_event == "sud.mg.merchant.game.process":
        return "sud_game_process"
    if notify_event.endswith(".game.process"):
        return "sud_game_process"
    if "settle" in notify_event.lower():
        return "sud_game_settle"
    return "sud_notify"


async def user_info_from_token(token: str) -> SudPlayerInfo:
    payload = decode_token(token)
    uid = str(payload.get("uid") or "")
    if uid.startswith("agent:"):
        agent_id = uid.removeprefix("agent:")
        difficulty = str(payload.get("difficulty") or "newbie")
        return await build_ai_player(agent_id, difficulty)
    return await build_user_player(uid)


async def user_info_from_code(code: str) -> tuple[str, datetime, SudPlayerInfo, dict[str, Any]]:
    payload = decode_token(code)
    uid = str(payload["uid"])
    session_id = payload.get("session_id")
    room_id = payload.get("room_id")
    ss_token, expires_at = make_ss_token(uid=uid, session_id=session_id, room_id=room_id)
    return ss_token, expires_at, await build_user_player(uid), payload


async def refresh_ss_token(token: str) -> tuple[str, datetime, SudPlayerInfo, dict[str, Any]]:
    payload = decode_token(token)
    uid = str(payload["uid"])
    session_id = payload.get("session_id")
    room_id = payload.get("room_id")
    ss_token, expires_at = make_ss_token(uid=uid, session_id=session_id, room_id=room_id)
    return ss_token, expires_at, await user_info_from_token(ss_token), payload


async def _append_event(
    *,
    session_id: str,
    event_type: str,
    state: str | None,
    payload: dict[str, Any],
    source: str,
    companion_reply: str | None,
) -> str:
    event_id = str(uuid.uuid4())
    await db.execute_raw(
        """
        INSERT INTO game_events (
            id, session_id, event_type, state, source, payload, companion_reply
        )
        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
        """,
        event_id,
        session_id,
        event_type,
        state,
        source,
        _json(payload),
        companion_reply,
    )
    return event_id


async def _write_game_message(
    *,
    conversation_id: str,
    role: str,
    content: str,
    metadata: dict[str, Any],
) -> str | None:
    try:
        created = await db.message.create(
            data={
                "conversation": {"connect": {"id": conversation_id}},
                "role": role,
                "content": content,
                "metadata": Json(metadata),
            }
        )
        return str(getattr(created, "id", "") or "")
    except Exception as exc:
        logger.warning("failed to persist game chat message: %r", exc)
        return None


async def _persist_game_status_to_chat_if_needed(
    previous: SudSessionResponse,
    updated: SudSessionResponse,
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
        if await _game_status_message_exists(updated.conversation_id, updated.id, status):
            return
        game_title = _game_title(updated, payload)
        actor_name = updated.ai_player.nick_name or "AI"
        text = f"{actor_name} 和你已{'退出' if status == 'ended' else '进入'}游戏《{game_title}》"
        metadata = {
            "kind": "game_status",
            "game_status": status,
            "game_title": game_title,
            "game_status_actor": "both",
            "game_status_actor_name": actor_name,
            "session_id": updated.id,
            "mg_id": updated.mg_id,
            "event_type": event_type,
            "state": state,
        }
        if status == "ended":
            metadata["game_ended_reason"] = _ended_reason(updated, event_type, state, payload)
        message_id = await _write_game_message(
            conversation_id=updated.conversation_id,
            role="assistant",
            content=text,
            metadata=metadata,
        )
        try:
            from app.services.runtime.ws_manager import manager

            await manager.send_event(
                updated.conversation_id,
                "game_status",
                {
                    "text": text,
                    "status": status,
                    "game_title": game_title,
                    "session_id": updated.id,
                    "mg_id": updated.mg_id,
                    "message_id": message_id or "",
                    "actor": "both",
                    "actor_name": actor_name,
                    "reason": metadata.get("game_ended_reason", ""),
                },
            )
        except Exception as exc:
            logger.debug("failed to emit game status websocket event: %r", exc)


async def _game_status_message_exists(
    conversation_id: str,
    session_id: str,
    status: str,
) -> bool:
    rows = await db.query_raw(
        """
        SELECT id
        FROM messages
        WHERE conversation_id = $1
          AND metadata->>'kind' = 'game_status'
          AND metadata->>'session_id' = $2
          AND metadata->>'game_status' = $3
        LIMIT 1
        """,
        conversation_id,
        session_id,
        status,
    )
    return bool(rows)


def _status_transition(
    previous: SudSessionResponse,
    updated: SudSessionResponse,
    event_type: str,
    state: str | None,
) -> str | None:
    if updated.status == "playing" and previous.status != "playing":
        return "started"
    if updated.status in _TERMINAL_STATUSES and previous.status not in _TERMINAL_STATUSES:
        return "ended"
    return None


def _game_title(session: SudSessionResponse, payload: dict[str, Any]) -> str:
    title = str(payload.get("game_title") or payload.get("gameName") or "").strip()
    if title:
        return title
    if session.mg_id == MONSTER_CRUSH_MG_ID:
        return "怪物消消乐"
    if _is_gomoku_session(session):
        return "五子棋"
    if session.mg_id == settings.sud_default_mg_id.strip():
        return "五子棋"
    return session.mg_id


def _ended_reason(
    session: SudSessionResponse,
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


async def _persist_reply_to_chat_if_needed(
    session: SudSessionResponse,
    event_type: str,
    state: str | None,
    reply: str | None,
) -> None:
    if not reply or not session.conversation_id:
        return
    if not _should_persist_reply_to_chat(event_type, state):
        return
    message_id = await _write_game_message(
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


async def _update_session_from_event(
    session: SudSessionResponse,
    event_type: str,
    state: str | None,
    payload: dict[str, Any],
    reply: str | None,
) -> None:
    status = session.status
    started_at = session.started_at
    ended_at = session.ended_at
    result = session.result
    duration_seconds = None

    if event_type in {"game_started", "sud_game_start"} or state == "mg_common_game_state":
        game_state = str(payload.get("gameState") or payload.get("game_state") or "")
        if event_type == "game_started" or game_state == "playing":
            status = "playing"
            started_at = started_at or _now().isoformat()

    if event_type in {"game_player_scores"} or state == "mg_common_game_player_scores":
        result = _merge_process_result(session, result, event_type, state, payload)

    if event_type in {"move", "sud_game_process", "sud_game_info", "game_process_info"}:
        result = _merge_process_result(session, result, event_type, state, payload)

    if event_type in {"game_settle", "sud_game_settle"} or state == "mg_common_game_settle":
        status = "settled"
        ended_at = _now().isoformat()
        result = _merge_process_result(
            session,
            _extract_result(session, payload),
            event_type,
            state,
            payload,
            previous_result=result,
        )
        duration_seconds = payload.get("battle_duration") or payload.get("duration")

    if (
        _is_abort_event(event_type, state, payload)
        and status not in _TERMINAL_STATUSES
        and (session.status == "playing" or started_at)
    ):
        status = "aborted"
        ended_at = _now().isoformat()
        result = _merge_process_result(session, result, event_type, state, payload)
        result = {
            **(result or {}),
            "game_round_id": payload.get("gameRoundId") or payload.get("game_round_id"),
            "room_id": payload.get("room_id") or session.room_id,
            "mg_id": payload.get("mg_id") or session.mg_id,
            "ended_reason": payload.get("reason") or payload.get("exit_reason") or "aborted",
            "user_outcome": "aborted",
        }

    await db.execute_raw(
        """
        UPDATE game_sessions
        SET
            status = $2,
            started_at = COALESCE($3::timestamptz, started_at),
            ended_at = COALESCE($4::timestamptz, ended_at),
            duration_seconds = COALESCE($5, duration_seconds),
            result = COALESCE($6::jsonb, result),
            companion_reply = COALESCE($7, companion_reply),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = $1
        """,
        session.id,
        status,
        started_at,
        ended_at,
        duration_seconds,
        _json(result) if result is not None else None,
        reply,
    )


def _row_to_session(row: dict[str, Any]) -> SudSessionResponse:
    user_player = SudPlayerInfo.model_validate(_loads(row.get("user_player"), {}))
    ai_player = SudPlayerInfo.model_validate(_loads(row.get("ai_player"), {}))
    return SudSessionResponse(
        id=str(row["id"]),
        status=str(row.get("status") or "created"),
        sdk_enabled=bool(row.get("sdk_enabled")),
        user_id=str(row.get("user_id") or ""),
        agent_id=str(row.get("agent_id") or ""),
        workspace_id=row.get("workspace_id"),
        conversation_id=row.get("conversation_id"),
        app_id=settings.sud_app_id.strip(),
        app_key=settings.sud_app_key.strip(),
        bundle_id=settings.sud_bundle_id.strip() or "com.companion.app",
        is_test_env=settings.sud_is_test_env,
        mg_id=str(row.get("mg_id") or settings.sud_default_mg_id or DEFAULT_DEMO_MG_ID),
        room_id=str(row.get("room_id") or ""),
        code=str(row.get("sud_code") or ""),
        code_expires_at=_iso(row.get("sud_code_expires_at")) or "",
        play_mode=str(row.get("play_mode") or "versus"),  # type: ignore[arg-type]
        difficulty=str(row.get("difficulty") or "newbie"),  # type: ignore[arg-type]
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


def _intro_reply(agent_name: str, play_mode: str, difficulty: str) -> str:
    if play_mode == "cooperate":
        return f"{agent_name}上线。我来和你一起配合通关，卡住的时候我会主动提醒，不抢你的节奏。"
    if difficulty == "hard":
        return f"{agent_name}上线。这局我会认真打，准备好了的话，我们就开局。"
    if difficulty == "normal":
        return f"{agent_name}上线。我会正常发挥，也会给你留一点观察空间。"
    return f"{agent_name}上线。我先陪你热身，新手局我会放慢一点，让你更容易看清节奏。"


def _reply_for_event(
    session: SudSessionResponse,
    event_type: str,
    state: str | None,
    payload: dict[str, Any],
) -> str | None:
    name = session.ai_player.nick_name
    if event_type == "sdk_ready":
        return f"{name}已经进房间了。我们先试一局，过程里的关键事件我都会记下来。"
    if event_type in {"game_started", "sud_game_start"}:
        if session.play_mode == "cooperate":
            return "开局。我会先观察局面，你需要我配合的时候我马上跟上。"
        if session.difficulty == "hard":
            return "开局。我这把不放水，看看谁先抓到机会。"
        return "开局啦。我会把节奏放缓一点，你大胆试。"
    if event_type == "move":
        if session.difficulty == "hard":
            return "这一步有点意思，我要认真防一下。"
        return "不错，这一步我看到了。你可以继续按这个方向试。"
    if event_type == "level_success":
        return "过了！这一步配合很顺，我把这局记录下来。"
    if event_type == "level_failed":
        return "没关系，我们换个思路再来。我会帮你盯住刚才卡住的位置。"
    if event_type in {"game_settle", "sud_game_settle"} or state == "mg_common_game_settle":
        result = _merge_process_result(
            session,
            _extract_result(session, payload),
            event_type,
            state,
            payload,
            previous_result=session.result,
        )
        outcome = result.get("user_outcome")
        process_text = _process_reply_fragment(result)
        if outcome == "win":
            return f"可以啊，这局你拿下了。{process_text}我刚才有点被你节奏带着跑，下局我得认真一点了。"
        if outcome == "lose":
            return f"哈哈，这把先算我小赢一局。{process_text}你不是乱输的，后面有几手已经很接近反超了。下局我陪你把那个节奏找回来。"
        return f"这局居然打平了。{process_text}感觉我们俩都没完全舒服起来，下一局可以换个更大胆的开局。"
    if _is_abort_event(event_type, state, payload):
        if session.status in _TERMINAL_STATUSES or session.status != "playing":
            return None
        title = str(payload.get("game_title") or "").strip()
        suffix = f"《{title}》" if title else "这局"
        return f"{suffix}先停在这里。我已经把进行到一半的分数和过程记录下来了，等你想继续玩的时候我们再接着来。"
    return None


def _should_persist_reply_to_chat(event_type: str, state: str | None) -> bool:
    if event_type == "game_settle":
        return False
    return (
        event_type
        in {
            "sud_game_settle",
            "level_success",
            "level_failed",
            "game_exited",
            "game_destroyed",
        }
        or state in {"mg_common_game_settle", "mg_common_destroy_game_scene"}
    )


def _is_abort_event(
    event_type: str,
    state: str | None,
    payload: dict[str, Any],
) -> bool:
    if event_type in {"game_exited", "game_destroyed"}:
        return True
    if state == "mg_common_destroy_game_scene":
        return True
    game_state = str(payload.get("gameState") or payload.get("game_state") or "").lower()
    return event_type == "sud_game_state" and game_state in {"destroyed", "closed", "aborted"}


def _extract_result(session: SudSessionResponse, payload: dict[str, Any]) -> dict[str, Any]:
    raw_results = payload.get("results") or payload.get("report_msg", {}).get("results") or []
    if not isinstance(raw_results, list):
        raw_results = []

    def normalize_outcome(value: Any) -> str:
        try:
            code = int(value)
        except Exception:
            return "unknown"
        if code == 2:
            return "win"
        if code == 1:
            return "lose"
        if code == 3:
            return "draw"
        return "unknown"

    user_row = None
    ai_row = None
    for row in raw_results:
        if not isinstance(row, dict):
            continue
        uid = str(row.get("uid") or "")
        if uid == session.user_player.uid:
            user_row = row
        if uid == session.ai_player.uid:
            ai_row = row

    result = {
        "game_round_id": payload.get("gameRoundId") or payload.get("game_round_id"),
        "room_id": payload.get("room_id") or session.room_id,
        "mg_id": payload.get("mg_id") or payload.get("mg_id_str") or session.mg_id,
        "duration_seconds": payload.get("battle_duration") or payload.get("duration"),
        "user": user_row,
        "ai": ai_row,
        "user_extras": _extract_extras(user_row),
        "ai_extras": _extract_extras(ai_row),
        "game_over_reason": _extract_game_over_reason(payload),
        "user_outcome": normalize_outcome((user_row or {}).get("isWin", (user_row or {}).get("is_win"))),
    }
    if _is_gomoku_session(session):
        gomoku = _extract_gomoku_settlement(session, result, payload)
        if gomoku:
            result["gomoku"] = gomoku
    return result


def _extract_extras(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {}
    raw = row.get("extras") or row.get("extra")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            decoded = json.loads(raw)
            return decoded if isinstance(decoded, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _extract_game_over_reason(payload: dict[str, Any]) -> Any:
    direct = payload.get("gameOverReason") or payload.get("game_over_reason")
    if direct is not None:
        return direct
    extras = payload.get("extras")
    if isinstance(extras, str):
        try:
            extras = json.loads(extras)
        except json.JSONDecodeError:
            extras = {}
    if isinstance(extras, dict):
        return extras.get("gameOverReason") or extras.get("game_over_reason")
    return None


def _is_gomoku_session(session: SudSessionResponse) -> bool:
    default_mg_id = settings.sud_default_mg_id.strip()
    return session.mg_id == GOMOKU_MG_ID or (
        bool(default_mg_id) and session.mg_id == default_mg_id
    )


def _extract_gomoku_settlement(
    session: SudSessionResponse,
    result: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    user = result.get("user")
    ai = result.get("ai")
    user_extras = result.get("user_extras")
    ai_extras = result.get("ai_extras")
    gomoku: dict[str, Any] = {}
    move_count = _first_number(
        payload,
        ("moveCount", "move_count", "stepCount", "step_count", "roundCount", "round_count"),
    )
    if move_count is None:
        move_count = _first_number(user_extras, ("moveCount", "move_count", "stepCount", "step_count"))
    if move_count is not None:
        gomoku["move_count"] = int(move_count)
    winning_line = _extract_gomoku_line(payload) or _extract_gomoku_line(user_extras) or _extract_gomoku_line(ai_extras)
    if winning_line:
        gomoku["winning_line"] = winning_line
        gomoku["win_direction"] = _gomoku_line_direction(winning_line)
    winner_uid = _winner_uid_from_result(session, user, ai)
    if winner_uid:
        gomoku["winner_uid"] = winner_uid
        gomoku["winner"] = "user" if winner_uid == session.user_player.uid else "ai"
    last_move = _extract_gomoku_move(session, payload)
    if last_move:
        gomoku["last_move"] = last_move
    return gomoku


def _winner_uid_from_result(
    session: SudSessionResponse,
    user: Any,
    ai: Any,
) -> str | None:
    if isinstance(user, dict) and _int_or_zero(user.get("isWin") or user.get("is_win")) == 2:
        return session.user_player.uid
    if isinstance(ai, dict) and _int_or_zero(ai.get("isWin") or ai.get("is_win")) == 2:
        return session.ai_player.uid
    return None


def _extract_gomoku_line(source: Any) -> list[dict[str, int]]:
    if not isinstance(source, dict):
        return []
    raw = (
        source.get("winningLine")
        or source.get("winning_line")
        or source.get("winLine")
        or source.get("win_line")
        or source.get("line")
    )
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = []
    if not isinstance(raw, list):
        return []
    points: list[dict[str, int]] = []
    for item in raw:
        point = _normalize_gomoku_point(item)
        if point is not None:
            points.append(point)
    return points


def _normalize_gomoku_point(value: Any) -> dict[str, int] | None:
    if isinstance(value, dict):
        x = _first_number(value, ("x", "col", "column", "chessX", "chess_x"))
        y = _first_number(value, ("y", "row", "chessY", "chess_y"))
        index = _first_number(value, ("index", "move_index", "pos", "position"))
        if (x is None or y is None) and index is not None:
            x = int(index) % 15
            y = int(index) // 15
        if x is not None and y is not None:
            return {"x": int(x), "y": int(y)}
    if isinstance(value, (int, float)):
        index = int(value)
        return {"x": index % 15, "y": index // 15}
    return None


def _gomoku_line_direction(line: list[dict[str, int]]) -> str | None:
    if len(line) < 2:
        return None
    first = line[0]
    last = line[-1]
    dx = last["x"] - first["x"]
    dy = last["y"] - first["y"]
    if dy == 0:
        return "horizontal"
    if dx == 0:
        return "vertical"
    if abs(dx) == abs(dy):
        return "diagonal"
    return None


def _merge_gomoku_process(
    session: SudSessionResponse,
    process: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    gomoku = dict(process.get("gomoku") or {})
    move = _extract_gomoku_move(session, payload)
    if move:
        moves = list(gomoku.get("moves") or [])
        added = False
        if not _gomoku_move_exists(moves, move):
            move["turn"] = int(gomoku.get("move_count") or len(moves)) + 1
            moves.append(move)
            added = True
        gomoku["moves"] = moves[-80:]
        gomoku["move_count"] = len(moves)
        gomoku["last_move"] = move
        actor = move.get("actor")
        if added and actor in {"user", "ai"}:
            gomoku[f"{actor}_moves"] = int(gomoku.get(f"{actor}_moves") or 0) + 1
    raw_moves = payload.get("moves") or payload.get("steps") or payload.get("chessList") or payload.get("chess_list")
    if isinstance(raw_moves, list):
        for item in raw_moves:
            item_move = _extract_gomoku_move(session, item if isinstance(item, dict) else {"index": item})
            if item_move:
                moves = list(gomoku.get("moves") or [])
                if not _gomoku_move_exists(moves, item_move):
                    item_move["turn"] = len(moves) + 1
                    moves.append(item_move)
                    gomoku["moves"] = moves[-80:]
                    gomoku["move_count"] = len(moves)
                    gomoku["last_move"] = item_move
    winning_line = _extract_gomoku_line(payload)
    if winning_line:
        gomoku["winning_line"] = winning_line
        gomoku["win_direction"] = _gomoku_line_direction(winning_line)
    winner_uid = str(payload.get("winnerUid") or payload.get("winner_uid") or payload.get("winner") or "")
    if winner_uid:
        gomoku["winner_uid"] = winner_uid
        if winner_uid == session.user_player.uid:
            gomoku["winner"] = "user"
        elif winner_uid == session.ai_player.uid:
            gomoku["winner"] = "ai"
    if gomoku:
        process["gomoku"] = gomoku
    return process


def _extract_gomoku_move(
    session: SudSessionResponse,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    move_source = payload.get("move") if isinstance(payload.get("move"), dict) else payload
    point = _normalize_gomoku_point(move_source)
    if point is None:
        point = _normalize_gomoku_point(
            {
                "index": payload.get("move_index")
                or payload.get("pos")
                or payload.get("position")
                or payload.get("index")
            }
        )
    if point is None:
        return None
    uid = str(
        payload.get("uid")
        or payload.get("userId")
        or payload.get("user_id")
        or payload.get("playerId")
        or payload.get("player_id")
        or ""
    )
    piece = str(payload.get("piece") or payload.get("chess") or payload.get("color") or "").strip()
    actor = None
    if uid == session.user_player.uid:
        actor = "user"
    elif uid == session.ai_player.uid:
        actor = "ai"
    elif piece.upper() in {"X", "BLACK", "B", "1", "黑"}:
        actor = "user"
    elif piece.upper() in {"O", "WHITE", "W", "2", "白"}:
        actor = "ai"
    move = {**point}
    if uid:
        move["uid"] = uid
    if actor:
        move["actor"] = actor
    if piece:
        move["piece"] = piece
    return move


def _gomoku_move_exists(moves: list[Any], move: dict[str, Any]) -> bool:
    for item in moves:
        if not isinstance(item, dict):
            continue
        if item.get("x") == move.get("x") and item.get("y") == move.get("y"):
            return True
    return False


def _first_number(source: Any, keys: tuple[str, ...]) -> int | float | None:
    if not isinstance(source, dict):
        return None
    for key in keys:
        value = source.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except Exception:
            continue
        return int(number) if number.is_integer() else number
    return None


def _merge_process_result(
    session: SudSessionResponse,
    result: dict[str, Any] | None,
    event_type: str,
    state: str | None,
    payload: dict[str, Any],
    *,
    previous_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged = {**(previous_result or session.result or {}), **(result or {})}
    process = dict(merged.get("process") or {})
    if event_type == "game_player_scores" or state == "mg_common_game_player_scores":
        process = _merge_score_process(session, process, payload)
    if _is_gomoku_session(session):
        process = _merge_gomoku_process(session, process, payload)
    if event_type == "sud_game_process":
        events = list(process.get("events") or [])
        event_name = str(payload.get("event") or state or "").strip()
        if event_name:
            events.append(
                {
                    "event": event_name,
                    "players": payload.get("players") or [],
                    "results": payload.get("results") or [],
                    "at": _now().isoformat(),
                }
            )
            process["events"] = events[-20:]
    if process:
        merged["process"] = process
    return merged


def _merge_score_process(
    session: SudSessionResponse,
    process: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    raw_scores = payload.get("scores") or []
    if not isinstance(raw_scores, list):
        return process
    latest_scores = {
        str(item.get("uid")): int(item.get("score") or 0)
        for item in raw_scores
        if isinstance(item, dict) and item.get("uid") is not None
    }
    if not latest_scores:
        return process
    user_score = latest_scores.get(session.user_player.uid)
    ai_score = latest_scores.get(session.ai_player.uid)
    leader = None
    lead = 0
    if user_score is not None and ai_score is not None:
        lead = user_score - ai_score
        if lead > 0:
            leader = "user"
        elif lead < 0:
            leader = "ai"
        else:
            leader = "tie"
    previous_leader = process.get("last_leader")
    lead_changes = int(process.get("lead_changes") or 0)
    if leader and previous_leader and leader != previous_leader and "tie" not in {leader, previous_leader}:
        lead_changes += 1
    return {
        **process,
        "score_updates": int(process.get("score_updates") or 0) + 1,
        "latest_scores": latest_scores,
        "user_score": user_score,
        "ai_score": ai_score,
        "last_leader": leader or previous_leader,
        "lead_changes": lead_changes,
        "max_user_lead": max(int(process.get("max_user_lead") or 0), lead),
        "max_ai_lead": max(int(process.get("max_ai_lead") or 0), -lead),
        "peak_user_score": max(int(process.get("peak_user_score") or 0), user_score or 0),
        "peak_ai_score": max(int(process.get("peak_ai_score") or 0), ai_score or 0),
        "last_score_at": _now().isoformat(),
    }


def _process_reply_fragment(result: dict[str, Any]) -> str:
    process = result.get("process") if isinstance(result, dict) else None
    if not isinstance(process, dict):
        process = {}
    user_row = result.get("user")
    ai_row = result.get("ai")
    user_score = process.get("user_score") or _score_from_result_row(user_row)
    ai_score = process.get("ai_score") or _score_from_result_row(ai_row)
    observations: list[str] = []
    lead_changes = int(process.get("lead_changes") or 0)
    if lead_changes:
        observations.append("中间还来回翻过一次节奏")
    gomoku_observation = _friendly_gomoku_observation(result, process)
    if gomoku_observation:
        observations.append(gomoku_observation)
    user_extras = result.get("user_extras")
    if isinstance(user_extras, dict):
        good = user_extras.get("numGood")
        crazy = user_extras.get("numCrazy")
        perfect = user_extras.get("numPerfect")
        excellent = user_extras.get("numExcellent")
        highlight = _friendly_combo_highlight(
            good=good,
            perfect=perfect,
            excellent=excellent,
            crazy=crazy,
        )
        if highlight:
            observations.append(highlight)
    score_observation = _friendly_score_observation(
        user_score=user_score,
        ai_score=ai_score,
    )
    if score_observation:
        observations.append(score_observation)
    if not observations:
        return "这局手感我已经记下来了，"
    return "，".join(observations[:2]) + "。"


def _friendly_combo_highlight(
    *,
    good: Any,
    perfect: Any,
    excellent: Any,
    crazy: Any,
) -> str | None:
    perfect_count = _int_or_zero(perfect)
    excellent_count = _int_or_zero(excellent)
    crazy_count = _int_or_zero(crazy)
    good_count = _int_or_zero(good)
    if crazy_count:
        return "你刚才有一波连得挺凶"
    if excellent_count:
        return "有几手消得很漂亮"
    if perfect_count:
        return f"你那 {perfect_count} 个 Perfect 挺漂亮"
    if good_count >= 6:
        return "你后面手感其实慢慢起来了"
    if good_count:
        return "有几步选择是对的"
    return None


def _friendly_gomoku_observation(
    result: dict[str, Any],
    process: dict[str, Any],
) -> str | None:
    gomoku = process.get("gomoku")
    if not isinstance(gomoku, dict):
        gomoku = {}
    settled = result.get("gomoku")
    if isinstance(settled, dict):
        gomoku = {**gomoku, **settled}
    move_count = _int_or_zero(gomoku.get("move_count"))
    direction = gomoku.get("win_direction")
    if move_count >= 20:
        return "这盘拖到中后段才分出来，挺有来回"
    if move_count >= 10:
        return "这盘不是几手就崩的，前面有几步防得住"
    if direction == "diagonal":
        return "最后那条斜线其实藏得挺深"
    if direction == "horizontal":
        return "最后横向那条线收得很快"
    if direction == "vertical":
        return "最后竖线压下来那一下挺关键"
    last_move = gomoku.get("last_move")
    if isinstance(last_move, dict) and last_move:
        return "最后那手位置我记下来了，下次可以提前一拍防"
    return None


def _friendly_score_observation(
    *,
    user_score: int | None,
    ai_score: int | None,
) -> str | None:
    if user_score is None or ai_score is None:
        return None
    gap = abs(user_score - ai_score)
    high_score = max(user_score, ai_score)
    if gap == 0:
        return "分数咬得很死"
    if high_score and gap / high_score <= 0.18:
        return "分差其实很小"
    if gap <= 15000:
        return "分差也没被拉到离谱"
    if user_score > ai_score:
        return "你这局把分数压得挺稳"
    return "我只是中段多攒出了一点优势"


def _int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _score_from_result_row(row: Any) -> int | None:
    if not isinstance(row, dict):
        return None
    raw_score = row.get("score")
    try:
        return int(raw_score)
    except Exception:
        return None


def expire_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)
