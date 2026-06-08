from __future__ import annotations

from typing import Any

from app.models.music import MusicTrack, MusicTrackPayload
from app.services.llm.models import get_chat_model, invoke_text
from app.services.prompting.store import get_prompt_text


def card_from_track(
    track: MusicTrack | MusicTrackPayload,
    *,
    intent: str,
    source: str,
) -> dict[str, Any]:
    title = track.title.strip() or "一起听一首歌"
    artist = track.artist.strip() or "Jamendo"
    library = track.library.strip() or "focus"
    footer = "推荐给你" if intent == "recommend" else "邀请一起听"
    accent = getattr(track, "accent_a", "") or "#1f6fff"
    return {
        "version": 1,
        "type": "music_track",
        "title": title,
        "subtitle": artist,
        "body": _library_label(library),
        "footer": footer,
        "accent": accent,
        "payload": {
            "intent": intent,
            "source": source,
            "track": _track_to_payload_dict(track),
        },
    }


async def render_music_reply(
    prompt_key: str,
    *,
    user_name: str,
    ai_name: str,
    track: MusicTrackPayload,
    activity: str = "处理自己的事",
    personality_brief: str = "",
    scene_hint: str = "",
) -> str:
    tpl = await get_prompt_text(prompt_key)
    prompt = tpl.format(
        user_name=user_name or "你",
        ai_name=ai_name or "我",
        song_name=track.title,
        artist=track.artist or "Jamendo",
        activity=activity or "处理自己的事",
        current_song=track.title,
        current_artist=track.artist or "Jamendo",
        personality_brief=personality_brief or "温和自然",
        scene_hint=scene_hint or "轻量分享一首适合此刻的歌。",
    )
    text = (await invoke_text(get_chat_model(), prompt)).strip()
    return _clean_single_reply(text)


def _track_to_payload_dict(track: MusicTrack | MusicTrackPayload) -> dict[str, Any]:
    metadata = getattr(track, "metadata", None) or {}
    return {
        "id": track.id,
        "title": track.title,
        "artist": track.artist,
        "album": track.album,
        "library": track.library,
        "url": track.url,
        "duration_sec": track.duration_sec,
        "cover_key": track.cover_key,
        "accent_a": track.accent_a,
        "accent_b": track.accent_b,
        "source": track.source,
        "metadata": metadata,
    }


def _library_label(library: str) -> str:
    tail = library.split(".")[-1].replace("_", " ").replace("-", " ").strip()
    return f"{tail.title()} 频道" if tail else "音乐频道"


def _clean_single_reply(text: str) -> str:
    first = text.split("||", 1)[0].strip()
    first = " ".join(first.split())
    if len(first) > 90:
        first = first[:90].rstrip("，。,.!！?？ ") + "…"
    return first or "好呀，我们一起听。"
