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

logger = logging.getLogger(__name__)

PAUSE_EXIT_SECONDS = 60
DISCONNECT_EXIT_SECONDS = 90


async def persist_and_emit_music_status(
    *,
    conversation_id: str,
    status: str,
    track: MusicTrack | MusicTrackPayload | None = None,
    reason: str | None = None,
) -> str:
    """Persist a co-listening timeline status and push it to active clients."""
    normalized = "ended" if status == "ended" else "started"
    text = "已退出共听" if normalized == "ended" else "已进入共听"
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
    )
    return True


async def end_if_paused_after_timeout(
    *,
    user_id: str,
    agent_id: str,
    conversation_id: str,
    seconds: int = PAUSE_EXIT_SECONDS,
) -> None:
    await asyncio.sleep(seconds)
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
    await end_co_listening_with_notice(
        user_id=user_id,
        agent_id=agent_id,
        conversation_id=conversation_id,
        reason="connection_lost",
    )


def _is_agent_auto_listening(session: Any) -> bool:
    return getattr(session, "initiated_by", None) == "agent_auto"


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
