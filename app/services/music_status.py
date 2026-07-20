from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from prisma import Json

from app.db import db
from app.models.music import MusicTrack, MusicTrackPayload
from app.services import music
from app.services.llm.models import get_utility_model, invoke_text
from app.services.music_chat import render_music_reply
from app.services.prompting.utils import render_prompt
from app.services.runtime.tasks import fire_background
from app.services.runtime.ws_manager import manager
from app.services.schedule_domain.schedule import get_cached_schedule, get_current_status

logger = logging.getLogger(__name__)

PAUSE_EXIT_SECONDS = 60
DISCONNECT_EXIT_SECONDS = 90
AGENT_WAIT_EXIT_SECONDS = 60
MANUAL_TRACK_CHANGE_REPLY_SECONDS = 120
AUTO_TRACK_CHANGE_REPLY_SECONDS = 600

_MANUAL_TRACK_CHANGE_SOURCES = {"manual_next", "manual_previous"}
_AUTO_TRACK_CHANGE_SOURCES = {"auto_next"}
_TRACK_CHANGE_PROMPT_KEYS = {
    "music.track_changed_manual",
    "music.track_changed_auto",
}
_PAUSE_FOLLOWUP_SKIP_KEYWORDS = (
    "睡觉",
    "睡了",
    "睡啦",
    "睡去",
    "要睡",
    "准备睡",
    "先睡",
    "晚安",
    "好梦",
    "困了",
    "明天",
    "下次再",
    "不听了",
    "先不听",
    "去忙",
    "有事",
    "忙去了",
    "洗澡",
    "开会",
)


async def persist_and_emit_music_status(
    *,
    conversation_id: str,
    status: str,
    track: MusicTrack | MusicTrackPayload | None = None,
    reason: str | None = None,
    actor: str | None = None,
    actor_name: str | None = None,
) -> str:
    """Persist a co-listening timeline status and push it to active clients."""
    normalized = "ended" if status == "ended" else "started"
    actor_label = _actor_label(actor=actor, actor_name=actor_name)
    text = f"{actor_label}{'已退出共听' if normalized == 'ended' else '已加入共听'}"
    track_title = (track.title if track else "") or ""
    track_id = (track.id if track else "") or ""
    message_id = await _persist_assistant_message(
        conversation_id,
        text,
        metadata={
            "music_status": normalized,
            "music_track_title": track_title,
            "music_track_id": track_id,
            "music_co_listening": normalized == "started",
            "music_status_actor": actor or "",
            "music_status_actor_name": actor_name or "",
            **({"music_ended_reason": reason} if reason else {}),
        },
    )
    await manager.send_event(
        conversation_id,
        "music_status",
        {
            "text": text,
            "status": normalized,
            "track_title": track_title,
            "track_id": track_id,
            "message_id": message_id,
            "actor": actor or "",
            "actor_name": actor_name or "",
            **({"reason": reason} if reason else {}),
        },
    )
    return message_id


async def end_co_listening_with_notice(
    *,
    user_id: str,
    agent_id: str,
    conversation_id: str,
    reason: str,
    prompt_key: str | None = None,
    activity: str = "处理自己的事",
    user_name: str = "你",
    ai_name: str = "我",
    status_actor: str | None = None,
    status_actor_name: str | None = None,
) -> bool:
    ended = await music.end_co_listening(
        user_id=user_id,
        agent_id=agent_id,
        conversation_id=conversation_id,
        reason=reason,
    )
    if ended is None:
        return False
    if _is_agent_auto_listening(ended):
        return False
    if prompt_key and ended.track is not None:
        reply = await _render_exit_reply(
            prompt_key,
            user_name=user_name,
            ai_name=ai_name,
            activity=activity,
            track=ended.track,
        )
        assistant_message_id = await _persist_assistant_message(
            conversation_id,
            reply,
            metadata={
                "music_co_listening": False,
                "music_prompt_key": prompt_key,
                "music_ended_reason": reason,
            },
        )
        await manager.send_event(
            conversation_id,
            "reply",
            {
                "text": reply,
                "assistant_message_id": assistant_message_id,
                "music_co_listening": False,
            },
        )
    await persist_and_emit_music_status(
        conversation_id=conversation_id,
        status="ended",
        track=ended.track,
        reason=reason,
        actor=status_actor,
        actor_name=status_actor_name,
    )
    return True


async def reconcile_co_listening_for_status(
    *,
    user_id: str,
    agent_id: str,
    conversation_id: str,
    workspace_id: str | None,
    status_code: str,
    activity: str,
    ai_name: str,
    user_name: str = "你",
) -> music.MusicCoListeningResponse | None:
    """Apply schedule status changes to music co-listening state."""
    normalized = (status_code or "idle").strip() or "idle"
    current = await music.get_open_co_listening(conversation_id=conversation_id)
    if normalized == "idle":
        if (
            current
            and await _is_waiting_for_agent_join(
                current,
                conversation_id=conversation_id,
            )
            and current.track is not None
        ):
            if current.is_playing:
                joined = await music.start_co_listening(
                    user_id=user_id,
                    agent_id=agent_id,
                    conversation_id=conversation_id,
                    workspace_id=workspace_id,
                    payload=_track_to_payload(current.track),
                    initiated_by="user_joined",
                    status="active",
                    position_seconds=current.position_seconds,
                    is_playing=True,
                )
                await _emit_rendered_reply(
                    conversation_id=conversation_id,
                    prompt_key="music.agent_join_after_busy",
                    user_name=user_name,
                    ai_name=ai_name,
                    activity=activity,
                    track=current.track,
                    music_co_listening=True,
                )
                await persist_and_emit_music_status(
                    conversation_id=conversation_id,
                    status="started",
                    track=current.track,
                    actor="agent",
                    actor_name=ai_name,
                )
                return joined
            ended = await music.end_co_listening(
                user_id=user_id,
                agent_id=agent_id,
                conversation_id=conversation_id,
                reason="user_stopped_before_agent_join",
            )
            if ended and ended.track is not None:
                await persist_and_emit_music_status(
                    conversation_id=conversation_id,
                    status="ended",
                    track=ended.track,
                    reason="user_stopped_before_agent_join",
                    actor="user",
                )
                await _emit_rendered_reply(
                    conversation_id=conversation_id,
                    prompt_key="music.agent_late_missed",
                    user_name=user_name,
                    ai_name=ai_name,
                    activity=activity,
                    track=ended.track,
                    music_co_listening=False,
                )
            return None
        if current and await _has_agent_joined(
            current,
            conversation_id=conversation_id,
        ):
            return current if current.status == "active" else None
        missed = await music.get_recent_unacknowledged_user_music_stop(
            conversation_id=conversation_id,
        )
        if missed and missed.track is not None:
            await _emit_rendered_reply(
                conversation_id=conversation_id,
                prompt_key="music.agent_late_missed",
                user_name=user_name,
                ai_name=ai_name,
                activity=activity,
                track=missed.track,
                music_co_listening=False,
            )
            await music.mark_user_music_stop_acknowledged(
                conversation_id=conversation_id,
            )
        return current if current and current.status == "active" else None

    if current and await _has_agent_joined(
        current,
        conversation_id=conversation_id,
    ):
        if _is_agent_auto_listening(current):
            await music.end_co_listening(
                user_id=user_id,
                agent_id=agent_id,
                conversation_id=conversation_id,
                reason=f"ai_{normalized}",
            )
            return None
        await end_co_listening_with_notice(
            user_id=user_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            reason=f"ai_{normalized}",
            prompt_key="music.busy_exit",
            activity=activity or "处理自己的事",
            user_name=user_name,
            ai_name=ai_name,
            status_actor="agent",
            status_actor_name=ai_name,
        )
        return None
    return current


async def reconcile_co_listening_for_current_schedule(
    *,
    user_id: str,
    agent_id: str,
    conversation_id: str,
    workspace_id: str | None,
    user_name: str = "你",
) -> music.MusicCoListeningResponse | None:
    state = await get_agent_current_schedule_state(agent_id)
    return await reconcile_co_listening_for_status(
        user_id=user_id,
        agent_id=agent_id,
        conversation_id=conversation_id,
        workspace_id=workspace_id,
        status_code=state["status"],
        activity=state["activity"],
        ai_name=state["ai_name"],
        user_name=user_name,
    )


async def get_agent_current_schedule_state(agent_id: str) -> dict[str, str]:
    schedule = await get_cached_schedule(agent_id)
    status = get_current_status(schedule) if schedule else {
        "status": "idle",
        "activity": "自由时间",
    }
    return {
        "status": str(status.get("status") or "idle"),
        "activity": str(status.get("activity") or status.get("event") or "处理自己的事"),
        "ai_name": await _resolve_agent_name(agent_id),
    }


async def maybe_emit_track_change_reply(
    *,
    conversation_id: str,
    current_session: music.MusicCoListeningResponse,
    next_track: MusicTrack,
    change_source: str | None,
) -> bool:
    """Emit a lightweight AI reply when a real track change happens in co-listening."""
    normalized_source = (change_source or "sync").strip()
    if current_session.status != "active" or _is_agent_auto_listening(current_session):
        return False
    previous_track = current_session.track
    if previous_track is None or previous_track.id == next_track.id:
        return False
    if normalized_source in _MANUAL_TRACK_CHANGE_SOURCES:
        prompt_key = "music.track_changed_manual"
        throttle_seconds = MANUAL_TRACK_CHANGE_REPLY_SECONDS
    elif normalized_source in _AUTO_TRACK_CHANGE_SOURCES:
        prompt_key = "music.track_changed_auto"
        throttle_seconds = AUTO_TRACK_CHANGE_REPLY_SECONDS
    else:
        return False
    if await _recent_music_prompt_exists(
        conversation_id=conversation_id,
        prompt_keys=_TRACK_CHANGE_PROMPT_KEYS,
        seconds=throttle_seconds,
    ):
        return False
    await _emit_rendered_reply(
        conversation_id=conversation_id,
        prompt_key=prompt_key,
        user_name="你",
        ai_name="我",
        activity="一起听歌",
        track=next_track,
        music_co_listening=True,
    )
    return True


async def scan_music_schedule_transitions(limit: int = 100) -> None:
    rows = await db.query_raw(
        """
        SELECT
            m.conversation_id,
            m.user_id,
            m.agent_id,
            m.workspace_id,
            a.name AS agent_name
        FROM music_co_listening_sessions m
        JOIN ai_agents a ON a.id = m.agent_id
        WHERE m.status IN ('active', 'pending_agent', 'agent_waiting_user')
        ORDER BY m.updated_at DESC
        LIMIT $1
        """,
        limit,
    )
    for raw in rows or []:
        try:
            await reconcile_co_listening_for_current_schedule(
                user_id=str(raw.get("user_id")),
                agent_id=str(raw.get("agent_id")),
                conversation_id=str(raw.get("conversation_id")),
                workspace_id=raw.get("workspace_id"),
            )
        except Exception as exc:
            logger.debug("music schedule transition scan skipped row: %s", exc)


async def end_if_paused_after_timeout(
    *,
    user_id: str,
    agent_id: str,
    conversation_id: str,
    seconds: int = PAUSE_EXIT_SECONDS,
) -> None:
    await asyncio.sleep(seconds)
    current = await music.get_open_co_listening(conversation_id=conversation_id)
    if current is not None and current.status == "active":
        await begin_user_exit_waiting_with_notice(
            user_id=user_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            reason="user_pause_timeout",
            stale_seconds=seconds,
        )
        return
    reason = "user_pause_timeout_before_agent_join"
    ended = await music.end_paused_co_listening_if_stale(
        user_id=user_id,
        agent_id=agent_id,
        conversation_id=conversation_id,
        paused_seconds=seconds,
        reason=reason,
    )
    if ended is None:
        return
    if _is_agent_auto_listening(ended):
        return
    if current is None or current.status != "active":
        await persist_and_emit_music_status(
            conversation_id=conversation_id,
            status="ended",
            track=ended.track,
            reason=reason,
            actor="user",
        )
        return


async def begin_user_exit_waiting_with_notice(
    *,
    user_id: str,
    agent_id: str,
    conversation_id: str,
    reason: str,
    stale_seconds: int | None = None,
) -> bool:
    if stale_seconds is None:
        waiting = await music.move_active_co_listening_to_agent_waiting(
            user_id=user_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            reason=reason,
        )
    else:
        waiting = await music.move_paused_active_co_listening_to_agent_waiting_if_stale(
            user_id=user_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            paused_seconds=stale_seconds,
            reason=reason,
        )
    if waiting is None:
        return False
    if _is_agent_auto_listening(waiting):
        return False
    should_follow_up = False
    if waiting.track is not None:
        should_follow_up = await should_emit_user_pause_followup(
            conversation_id=conversation_id,
            track=waiting.track,
        )
    await persist_and_emit_music_status(
        conversation_id=conversation_id,
        status="ended",
        track=waiting.track,
        reason=reason,
        actor="user",
    )
    if waiting.track is not None and should_follow_up:
        reply = await _render_exit_reply(
            "music.user_pause_exit",
            user_name="你",
            ai_name="我",
            activity="等你继续听歌",
            track=waiting.track,
        )
        assistant_message_id = await _persist_assistant_message(
            conversation_id,
            reply,
            metadata={
                "music_co_listening": False,
                "music_prompt_key": "music.user_pause_exit",
                "music_ended_reason": reason,
            },
        )
        await manager.send_event(
            conversation_id,
            "reply",
            {
                "text": reply,
                "assistant_message_id": assistant_message_id,
                "music_co_listening": False,
            },
        )
    fire_background(
        end_agent_waiting_after_timeout(
            user_id=user_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
        )
    )
    return True


async def should_emit_user_pause_followup(
    *,
    conversation_id: str,
    track: MusicTrack,
) -> bool:
    """Decide whether a pause-timeout exit should ask why the user left.

    Pause timeout is only a signal. If recent chat already explains the stop
    (sleeping, saying good night, going busy, explicitly ending), sending a
    "where did you go" follow-up feels mechanical.
    """
    try:
        lines, recent_user_text = await _recent_non_status_chat_context(
            conversation_id=conversation_id,
            limit=8,
        )
    except Exception as exc:
        logger.debug("music pause follow-up context unavailable: %s", exc)
        return True
    if not lines:
        return True
    if _has_clear_pause_reason(recent_user_text):
        return False
    try:
        raw = await render_prompt(
            "music.user_pause_followup_decision",
            {
                "recent_context": "\n".join(lines),
                "song_name": track.title,
                "artist": track.artist or "Jamendo",
            },
            lambda p: invoke_text(get_utility_model(), p),
            strip_split=False,
        )
    except Exception as exc:
        logger.debug("music pause follow-up decision failed: %s", exc)
        return True
    decision = _parse_pause_followup_decision(str(raw or ""))
    return True if decision is None else decision


async def end_agent_waiting_after_timeout(
    *,
    user_id: str,
    agent_id: str,
    conversation_id: str,
    seconds: int = AGENT_WAIT_EXIT_SECONDS,
) -> None:
    await asyncio.sleep(seconds)
    ended = await music.end_agent_waiting_if_stale(
        user_id=user_id,
        agent_id=agent_id,
        conversation_id=conversation_id,
        waiting_seconds=seconds,
        reason="user_absent_timeout",
    )
    if ended is None or _is_agent_auto_listening(ended):
        return
    if ended.track is not None:
        reply = await _render_exit_reply(
            "music.user_absent_exit",
            user_name="你",
            ai_name="我",
            activity="等你继续听歌",
            track=ended.track,
        )
        assistant_message_id = await _persist_assistant_message(
            conversation_id,
            reply,
            metadata={
                "music_co_listening": False,
                "music_prompt_key": "music.user_absent_exit",
                "music_ended_reason": "user_absent_timeout",
            },
        )
        await manager.send_event(
            conversation_id,
            "reply",
            {
                "text": reply,
                "assistant_message_id": assistant_message_id,
                "music_co_listening": False,
            },
        )
    await persist_and_emit_music_status(
        conversation_id=conversation_id,
        status="ended",
        track=ended.track,
        reason="user_absent_timeout",
        actor="agent",
        actor_name=await _resolve_agent_name(agent_id),
    )


async def end_if_disconnected_after_timeout(
    *,
    user_id: str,
    agent_id: str,
    conversation_id: str,
    seconds: int = DISCONNECT_EXIT_SECONDS,
) -> None:
    await asyncio.sleep(seconds)
    if manager.get(conversation_id) is not None:
        return
    current = await music.get_open_co_listening(conversation_id=conversation_id)
    if current is not None and current.status == "active":
        await begin_user_exit_waiting_with_notice(
            user_id=user_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            reason="connection_lost",
        )
        return
    if current is not None and current.status == "agent_waiting_user":
        return
    if current is None or current.status != "active":
        if current is not None:
            ended = await music.end_co_listening(
                user_id=user_id,
                agent_id=agent_id,
                conversation_id=conversation_id,
                reason="connection_lost_before_agent_join",
            )
            if ended is not None and not _is_agent_auto_listening(ended):
                await persist_and_emit_music_status(
                    conversation_id=conversation_id,
                    status="ended",
                    track=ended.track,
                    reason="connection_lost_before_agent_join",
                    actor="user",
                )
        return
    await end_co_listening_with_notice(
        user_id=user_id,
        agent_id=agent_id,
        conversation_id=conversation_id,
        reason="connection_lost",
        prompt_key="music.user_pause_exit",
        status_actor="user",
    )


def _is_agent_auto_listening(session: Any) -> bool:
    return getattr(session, "initiated_by", None) == "agent_auto"


async def _has_agent_joined(session: Any, *, conversation_id: str) -> bool:
    status = getattr(session, "status", None)
    if status not in {"active", "agent_waiting_user"}:
        return False
    initiated_by = getattr(session, "initiated_by", None)
    if initiated_by == "user_pending":
        return False
    if initiated_by in {"agent", "agent_auto", "user_joined"}:
        return True
    if initiated_by == "user":
        return await _agent_join_status_exists(conversation_id=conversation_id)
    return True


async def _is_waiting_for_agent_join(session: Any, *, conversation_id: str) -> bool:
    if getattr(session, "status", None) == "pending_agent":
        return True
    if getattr(session, "status", None) != "active":
        return False
    initiated_by = getattr(session, "initiated_by", None)
    if initiated_by == "user_pending":
        return True
    if initiated_by == "user_joined":
        return False
    if initiated_by == "user":
        return not await _agent_join_status_exists(conversation_id=conversation_id)
    return False


async def _agent_join_status_exists(*, conversation_id: str) -> bool:
    rows = await db.query_raw(
        """
        SELECT id
        FROM messages
        WHERE conversation_id = $1
          AND role = 'assistant'
          AND metadata ->> 'music_status' = 'started'
          AND metadata ->> 'music_status_actor' = 'agent'
        LIMIT 1
        """,
        conversation_id,
    )
    return bool(rows)


def _actor_label(*, actor: str | None, actor_name: str | None) -> str:
    if actor == "user":
        return "你"
    if actor == "agent":
        return (actor_name or "对方").strip() or "对方"
    return ""


async def _recent_music_prompt_exists(
    *,
    conversation_id: str,
    prompt_keys: set[str],
    seconds: int,
) -> bool:
    if not prompt_keys:
        return False
    rows = await db.query_raw(
        """
        SELECT id
        FROM messages
        WHERE conversation_id = $1
          AND role = 'assistant'
          AND metadata ->> 'music_prompt_key' = ANY($2::text[])
          AND created_at >= now() - make_interval(secs => $3::int)
        LIMIT 1
        """,
        conversation_id,
        list(prompt_keys),
        seconds,
    )
    return bool(rows)


async def _recent_non_status_chat_context(
    *,
    conversation_id: str,
    limit: int,
) -> tuple[list[str], str]:
    rows = await db.query_raw(
        """
        SELECT role, content
        FROM messages
        WHERE conversation_id = $1
          AND content IS NOT NULL
          AND content <> ''
          AND (
              metadata IS NULL
              OR metadata ->> 'music_status' IS NULL
          )
        ORDER BY created_at DESC
        LIMIT $2
        """,
        conversation_id,
        limit,
    )
    normalized = list(reversed([_message_row(row) for row in rows]))
    lines: list[str] = []
    user_parts: list[str] = []
    for row in normalized:
        role = str(row.get("role") or "").strip()
        content = " ".join(str(row.get("content") or "").split())
        if not content:
            continue
        if len(content) > 120:
            content = content[:120].rstrip() + "…"
        label = "用户" if role == "user" else "AI"
        lines.append(f"{label}: {content}")
        if role == "user":
            user_parts.append(content)
    return lines, "\n".join(user_parts[-4:])


def _message_row(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return row
    try:
        return dict(row)
    except Exception:
        return {}


def _has_clear_pause_reason(user_text: str) -> bool:
    compact = "".join((user_text or "").lower().split())
    if not compact:
        return False
    return any(keyword in compact for keyword in _PAUSE_FOLLOWUP_SKIP_KEYWORDS)


def _parse_pause_followup_decision(raw: str) -> bool | None:
    text = (raw or "").strip()
    if not text:
        return None
    if "{" in text and "}" in text:
        text = text[text.index("{") : text.rindex("}") + 1]
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    value = data.get("should_ask")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "ask", "需要", "追问"}:
            return True
        if normalized in {"false", "no", "0", "skip", "不需要", "不追问"}:
            return False
    return None


async def _emit_rendered_reply(
    *,
    conversation_id: str,
    prompt_key: str,
    user_name: str,
    ai_name: str,
    activity: str,
    track: MusicTrack,
    music_co_listening: bool,
) -> str:
    # trace 包装 (测试手册 §4 缺口): 音乐事件回复也带 Trace 按钮,
    # 运维可在面板内调试 music.busy_exit / agent_join_after_busy 等 prompt.
    from app.services.llm.usage_tracker import traced_usage_session

    async with traced_usage_session(
        name=f"[music:{prompt_key}]", scope="music",
        conversation_id=conversation_id, agent_id=None, user_id=None,
    ) as tracer:
        reply = await _render_exit_reply(
            prompt_key,
            user_name=user_name,
            ai_name=ai_name,
            activity=activity,
            track=track,
        )
        metadata = {
            "music_co_listening": music_co_listening,
            "music_prompt_key": prompt_key,
        }
        if tracer.safe_trace_id:
            metadata["trace_id"] = tracer.safe_trace_id
        assistant_message_id = await _persist_assistant_message(
            conversation_id,
            reply,
            metadata=metadata,
        )
    await manager.send_event(
        conversation_id,
        "reply",
        {
            "text": reply,
            "assistant_message_id": assistant_message_id,
            "music_co_listening": music_co_listening,
        },
    )
    return assistant_message_id


async def _render_exit_reply(
    prompt_key: str,
    *,
    user_name: str,
    ai_name: str,
    activity: str,
    track: MusicTrack,
) -> str:
    try:
        return await render_music_reply(
            prompt_key,
            user_name=user_name,
            ai_name=ai_name,
            track=_track_to_payload(track),
            activity=activity,
        )
    except Exception as exc:
        logger.warning("music exit reply generation failed: %s", exc)
        if prompt_key == "music.user_pause_exit":
            return "你怎么停了呀，是去忙什么了吗？"
        if prompt_key == "music.user_absent_exit":
            return "你看起来去忙了，那我们下次再一起听。"
        if prompt_key == "music.switch_track":
            return f"切到《{track.title}》啦，我们继续听。"
        if prompt_key == "music.track_changed_manual":
            return f"切到《{track.title}》啦，这首我们继续听。"
        if prompt_key == "music.track_changed_auto":
            return f"下一首是《{track.title}》，感觉可以接着听下去。"
        if prompt_key == "music.busy_exit":
            return f"我得先去{activity}了，有点可惜，下次我们继续听。"
        if prompt_key == "music.agent_join_after_busy":
            return f"我刚忙完回来啦，现在可以一起听《{track.title}》了。"
        if prompt_key == "music.agent_late_missed":
            return f"抱歉我来晚了，看到你已经不听《{track.title}》了，下次我们再一起听。"
        return "先暂停一起听啦，等会儿再接着。"


async def _resolve_agent_name(agent_id: str) -> str:
    try:
        agent = await db.aiagent.find_unique(where={"id": agent_id})
    except Exception:
        return "对方"
    return (getattr(agent, "name", None) or "对方").strip() or "对方"


async def _persist_assistant_message(
    conversation_id: str,
    text: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> str:
    # 系统标记收口: 音乐消息 prompt 带 reply_prefix, 单独剥 [EMO:]/条数标记.
    # 剥完为空 (整条都是标记) 用占位省略号, 绝不回退未清理原文.
    from app.services.chat.reply_formatting import strip_system_markers

    text = strip_system_markers(text) or "..."
    created = await db.message.create(
        data={
            "conversation": {"connect": {"id": conversation_id}},
            "role": "assistant",
            "content": text,
            **({"metadata": Json(metadata)} if metadata else {}),
        }
    )
    return created.id


def _track_to_payload(track: MusicTrack) -> MusicTrackPayload:
    return MusicTrackPayload(
        id=track.id,
        title=track.title,
        artist=track.artist,
        album=track.album,
        library=track.library,
        url=track.url,
        duration_sec=track.duration_sec,
        cover_key=track.cover_key,
        accent_a=track.accent_a,
        accent_b=track.accent_b,
        source=track.source,
        metadata=track.metadata,
    )
