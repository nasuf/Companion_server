from __future__ import annotations

import asyncio
import logging
from typing import Any

from prisma import Json

from app.db import db
from app.models.music import MusicTrack, MusicTrackPayload
from app.services import music
from app.services.music_chat import render_music_reply
from app.services.runtime.ws_manager import manager
from app.services.schedule_domain.schedule import get_cached_schedule, get_current_status

logger = logging.getLogger(__name__)

PAUSE_EXIT_SECONDS = 60
DISCONNECT_EXIT_SECONDS = 90


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
    text = f"{actor_label}{'已退出共听' if normalized == 'ended' else '已进入共听'}"
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
        if current and current.status == "pending_agent" and current.track is not None:
            if current.is_playing:
                joined = await music.start_co_listening(
                    user_id=user_id,
                    agent_id=agent_id,
                    conversation_id=conversation_id,
                    workspace_id=workspace_id,
                    payload=_track_to_payload(current.track),
                    initiated_by="user",
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

    if current and current.status == "active":
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
        WHERE m.status IN ('active', 'pending_agent')
        ORDER BY m.updated_at DESC
        LIMIT $1
        """,
        limit,
    )
    for raw in rows or []:
        try:
            schedule = await get_cached_schedule(str(raw.get("agent_id")))
            status = get_current_status(schedule) if schedule else {
                "status": "idle",
                "activity": "自由时间",
            }
            await reconcile_co_listening_for_status(
                user_id=str(raw.get("user_id")),
                agent_id=str(raw.get("agent_id")),
                conversation_id=str(raw.get("conversation_id")),
                workspace_id=raw.get("workspace_id"),
                status_code=str(status.get("status") or "idle"),
                activity=str(status.get("activity") or status.get("event") or "处理自己的事"),
                ai_name=str(raw.get("agent_name") or "我"),
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
    ended = await music.end_paused_co_listening_if_stale(
        user_id=user_id,
        agent_id=agent_id,
        conversation_id=conversation_id,
        paused_seconds=seconds,
        reason="user_pause_timeout",
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
            reason="user_pause_timeout",
            actor="user",
        )
        return
    if ended.track is not None:
        reply = await _render_exit_reply(
            "music.user_pause_exit",
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
                "music_prompt_key": "music.user_pause_exit",
                "music_ended_reason": "user_pause_timeout",
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
        reason="user_pause_timeout",
        actor="user",
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
    if current is None or current.status != "active":
        if current is not None:
            await music.end_co_listening(
                user_id=user_id,
                agent_id=agent_id,
                conversation_id=conversation_id,
                reason="connection_lost",
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


def _actor_label(*, actor: str | None, actor_name: str | None) -> str:
    if actor == "user":
        return "你"
    if actor == "agent":
        return (actor_name or "对方").strip() or "对方"
    return ""


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
    reply = await _render_exit_reply(
        prompt_key,
        user_name=user_name,
        ai_name=ai_name,
        activity=activity,
        track=track,
    )
    assistant_message_id = await _persist_assistant_message(
        conversation_id,
        reply,
        metadata={
            "music_co_listening": music_co_listening,
            "music_prompt_key": prompt_key,
        },
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
        if prompt_key == "music.busy_exit":
            return f"我得先去{activity}了，有点可惜，下次我们继续听。"
        if prompt_key == "music.agent_join_after_busy":
            return f"我刚忙完回来啦，现在可以一起听《{track.title}》了。"
        if prompt_key == "music.agent_late_missed":
            return f"抱歉我来晚了，看到你已经不听《{track.title}》了，下次我们再一起听。"
        return "先暂停一起听啦，等会儿再接着。"


async def _persist_assistant_message(
    conversation_id: str,
    text: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> str:
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
