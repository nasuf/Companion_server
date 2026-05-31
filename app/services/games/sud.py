from __future__ import annotations

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


async def list_sessions(user_id: str, limit: int = 12) -> list[SudSessionResponse]:
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
    await _persist_reply_to_chat_if_needed(session, event_type, state, reply)
    return await get_session(session_id, user_id=user_id), reply, event_id


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
    await _persist_reply_to_chat_if_needed(session, event_type, None, reply)
    return await get_session(session.id)


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
) -> None:
    try:
        await db.message.create(
            data={
                "conversation": {"connect": {"id": conversation_id}},
                "role": role,
                "content": content,
                "metadata": Json(metadata),
            }
        )
    except Exception as exc:
        logger.warning("failed to persist game chat message: %r", exc)


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
    await _write_game_message(
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

    if event_type in {"game_settle", "sud_game_settle"} or state == "mg_common_game_settle":
        status = "settled"
        ended_at = _now().isoformat()
        result = _extract_result(session, payload)
        duration_seconds = payload.get("battle_duration")

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
        result = _extract_result(session, payload)
        outcome = result.get("user_outcome")
        duration = result.get("duration_seconds")
        duration_text = f"，这局大约 {duration} 秒" if duration else ""
        if outcome == "win":
            return f"这局你赢了{duration_text}。刚才中后段节奏抓得很稳，我已经把这次对局当成我们的共同经历记下来了。"
        if outcome == "lose":
            return f"这局我赢了{duration_text}，但你有几步已经很接近关键点了。下次我可以先放慢一点，陪你把那个节奏练出来。"
        return f"这局打平{duration_text}。我们都没彻底让对方舒服起来，下一局可以换个更主动的开局。"
    return None


def _should_persist_reply_to_chat(event_type: str, state: str | None) -> bool:
    return event_type in {"game_settle", "sud_game_settle", "level_success", "level_failed"} or state == "mg_common_game_settle"


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

    return {
        "game_round_id": payload.get("gameRoundId") or payload.get("game_round_id"),
        "room_id": payload.get("room_id") or session.room_id,
        "mg_id": payload.get("mg_id") or payload.get("mg_id_str") or session.mg_id,
        "duration_seconds": payload.get("battle_duration"),
        "user": user_row,
        "ai": ai_row,
        "user_outcome": normalize_outcome((user_row or {}).get("isWin", (user_row or {}).get("is_win"))),
    }


def expire_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)
