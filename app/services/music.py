from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any

import httpx

from app.config import settings
from app.db import db
from app.models.music import MusicLibrary, MusicTrack, MusicTrackPayload

logger = logging.getLogger(__name__)

AUDIOLIB_AUDIO_PATH = "/v1/audio"
DEFAULT_LIBRARIES = ["audio.focus", "audio.relax", "audio.sleep"]
DEFAULT_TRACK_LIMIT = 1
AUDIO_CACHE_TTL_SECONDS = 5 * 60
_TIMEOUT = httpx.Timeout(8.0, connect=4.0)
_audio_cache: dict[tuple[str, str, str, int], tuple[float, MusicTrack]] = {}
_LIBRARY_CATALOG = {
    "audio.default": ("默认", "轻松随机"),
    "audio.focus": ("专注", "工作和阅读"),
    "audio.relax": ("放松", "慢下来"),
    "audio.sleep": ("睡眠", "夜间陪伴"),
    "audio.background": ("背景", "轻量氛围"),
    "audio.workout": ("运动", "有氧节奏"),
    "audio.energy": ("能量", "提神一点"),
}
_ACCENTS = [
    ("#1f6fff", "#18c6c0"),
    ("#7c3cff", "#1f6fff"),
    ("#ff8a3d", "#7c3cff"),
    ("#22c66b", "#1f6fff"),
    ("#18c6c0", "#22c66b"),
    ("#101820", "#1f6fff"),
]
_MOCK_TRACKS = [
    ("云烟成雨", "房东的猫", "云烟成雨 - Single", 238),
    ("夜空中最亮的星", "逃跑计划", "世界", 274),
    ("给你一瓶魔法药水", "告五人", "玫瑰凭证", 253),
    ("慢慢喜欢你", "莫文蔚", "慢慢喜欢你 - Single", 241),
    ("玫瑰少年", "五月天", "玫瑰少年 - Single", 230),
    ("如果可以", "韦礼安", "如果可以 - From THE FIRST TAKE", 286),
]


def api_enabled() -> bool:
    return bool(settings.audiolib_api_key.strip())


def missing_config() -> list[str]:
    return [] if api_enabled() else ["AUDIOLIB_API_KEY"]


def default_libraries() -> list[str]:
    raw = settings.audiolib_default_libraries.strip()
    if not raw:
        return DEFAULT_LIBRARIES
    libraries = [item.strip() for item in raw.split(",") if item.strip()]
    return libraries or DEFAULT_LIBRARIES


def libraries() -> list[MusicLibrary]:
    configured = default_libraries()
    ordered_ids = configured + [
        library_id for library_id in _LIBRARY_CATALOG if library_id not in configured
    ]
    return [
        MusicLibrary(
            id=library_id,
            title=_LIBRARY_CATALOG.get(library_id, (_library_label(library_id), ""))[0],
            subtitle=_LIBRARY_CATALOG.get(library_id, ("", ""))[1],
        )
        for library_id in ordered_ids
    ]


async def ensure_agent_owner(user_id: str, agent_id: str) -> dict[str, Any]:
    rows = await db.query_raw(
        """
        SELECT id, name, user_id
        FROM ai_agents
        WHERE id = $1 AND user_id = $2 AND status <> 'archived'
        LIMIT 1
        """,
        agent_id,
        user_id,
    )
    if not rows:
        raise ValueError("agent_not_found")
    return _row(rows[0])


async def resolve_workspace(
    *,
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
) -> str | None:
    if not workspace_id:
        return None
    rows = await db.query_raw(
        """
        SELECT id
        FROM chat_workspaces
        WHERE id = $1 AND user_id = $2 AND agent_id = $3 AND status = 'active'
        LIMIT 1
        """,
        workspace_id,
        user_id,
        agent_id,
    )
    if not rows:
        raise ValueError("workspace_not_found")
    return workspace_id


async def list_square_tracks(
    *,
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
    library: str | None,
    limit: int,
    refresh: bool = False,
) -> list[MusicTrack]:
    await ensure_agent_owner(user_id, agent_id)
    await resolve_workspace(
        user_id=user_id,
        agent_id=agent_id,
        workspace_id=workspace_id,
    )
    favorites = await favorite_ids(user_id=user_id, agent_id=agent_id)
    libraries = [library] if library else default_libraries()
    tracks: list[MusicTrack] = []
    for index in range(max(1, limit)):
        chosen = libraries[index % len(libraries)]
        track = await fetch_random_track(chosen, index=index, use_cache=not refresh)
        tracks.append(track.model_copy(update={"is_favorite": track.id in favorites}))
    return tracks


async def fetch_random_track(
    library: str,
    *,
    index: int = 0,
    use_cache: bool = True,
) -> MusicTrack:
    if not api_enabled():
        return _mock_track(library, index=index)
    base_url = settings.audiolib_base_url.strip().rstrip("/") or "https://api.audiolib.ai"
    api_key = settings.audiolib_api_key.strip()
    cache_key = (_track_id(api_key), base_url, library, index)
    cached = _audio_cache.get(cache_key)
    now = time.monotonic()
    if use_cache and cached and now - cached[0] < AUDIO_CACHE_TTL_SECONDS:
        return cached[1].model_copy()
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, trust_env=False) as client:
            response = await client.post(
                f"{base_url}{AUDIOLIB_AUDIO_PATH}",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"library": library},
            )
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("AudioLib request failed, using mock music: %s", exc)
        return _mock_track(library, index=index)
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        logger.warning("AudioLib returned unexpected response, using mock music")
        return _mock_track(library, index=index)
    title = _clean_text(data.get("title")) or "Untitled Audio"
    url = _clean_text(data.get("url"))
    duration = _int(data.get("duration_sec"))
    track_id = _track_id(url or f"{library}:{title}")
    visual_index = _visual_index(track_id)
    accent_a, accent_b = _ACCENTS[visual_index % len(_ACCENTS)]
    track = MusicTrack(
        id=track_id,
        title=title,
        artist="AudioLib",
        album=_library_title(library),
        library=library,
        url=url,
        duration_sec=duration,
        cover_key=f"music-cover-{(visual_index % 11) + 1:02d}.jpg",
        accent_a=accent_a,
        accent_b=accent_b,
        metadata={"raw": data},
    )
    _audio_cache[cache_key] = (now, track)
    return track


async def favorite_ids(*, user_id: str, agent_id: str) -> set[str]:
    rows = await db.query_raw(
        """
        SELECT track_external_id
        FROM music_favorites
        WHERE user_id = $1 AND agent_id = $2
        """,
        user_id,
        agent_id,
    )
    return {_row(row).get("track_external_id", "") for row in rows}


async def list_favorites(*, user_id: str, agent_id: str) -> list[MusicTrack]:
    await ensure_agent_owner(user_id, agent_id)
    rows = await db.query_raw(
        """
        SELECT *
        FROM music_favorites
        WHERE user_id = $1 AND agent_id = $2
        ORDER BY created_at DESC
        """,
        user_id,
        agent_id,
    )
    return [_favorite_row_to_track(_row(row)) for row in rows]


async def add_favorite(
    *,
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
    payload: MusicTrackPayload,
) -> MusicTrack:
    await ensure_agent_owner(user_id, agent_id)
    workspace_id = await resolve_workspace(
        user_id=user_id,
        agent_id=agent_id,
        workspace_id=workspace_id,
    )
    track = _payload_to_track(payload, is_favorite=True)
    await db.execute_raw(
        """
        INSERT INTO music_favorites (
            id, user_id, agent_id, workspace_id, track_external_id, title, artist,
            album, library, audio_url, duration_sec, cover_key, accent_a,
            accent_b, source, metadata
        )
        VALUES (
            $1, $2, $3, $4, $5, $6, $7,
            $8, $9, $10, $11, $12, $13,
            $14, $15, $16::jsonb
        )
        ON CONFLICT (user_id, agent_id, track_external_id)
        DO UPDATE SET
            workspace_id = EXCLUDED.workspace_id,
            title = EXCLUDED.title,
            artist = EXCLUDED.artist,
            album = EXCLUDED.album,
            library = EXCLUDED.library,
            audio_url = EXCLUDED.audio_url,
            duration_sec = EXCLUDED.duration_sec,
            cover_key = EXCLUDED.cover_key,
            accent_a = EXCLUDED.accent_a,
            accent_b = EXCLUDED.accent_b,
            source = EXCLUDED.source,
            metadata = EXCLUDED.metadata,
            updated_at = now()
        """,
        str(uuid.uuid4()),
        user_id,
        agent_id,
        workspace_id,
        track.id,
        track.title,
        track.artist,
        track.album,
        track.library,
        track.url,
        track.duration_sec,
        track.cover_key,
        track.accent_a,
        track.accent_b,
        track.source,
        json.dumps(track.metadata, ensure_ascii=False),
    )
    return track


async def remove_favorite(*, user_id: str, agent_id: str, track_id: str) -> bool:
    await ensure_agent_owner(user_id, agent_id)
    count = await db.execute_raw(
        """
        DELETE FROM music_favorites
        WHERE user_id = $1 AND agent_id = $2 AND track_external_id = $3
        """,
        user_id,
        agent_id,
        track_id,
    )
    return bool(count)


async def get_now_playing(*, user_id: str, agent_id: str) -> tuple[MusicTrack | None, int, bool, str | None]:
    await ensure_agent_owner(user_id, agent_id)
    favorites = await favorite_ids(user_id=user_id, agent_id=agent_id)
    rows = await db.query_raw(
        """
        SELECT *
        FROM music_playbacks
        WHERE user_id = $1 AND agent_id = $2
        LIMIT 1
        """,
        user_id,
        agent_id,
    )
    if not rows:
        return None, 0, False, None
    row = _row(rows[0])
    track = _playback_row_to_track(row).model_copy(
        update={"is_favorite": row.get("track_external_id") in favorites}
    )
    return (
        track,
        _int(row.get("position_seconds")),
        bool(row.get("is_playing")),
        _iso(row.get("updated_at")),
    )


async def upsert_now_playing(
    *,
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
    payload: MusicTrackPayload,
    position_seconds: int,
    is_playing: bool,
) -> tuple[MusicTrack, int, bool, str | None]:
    await ensure_agent_owner(user_id, agent_id)
    workspace_id = await resolve_workspace(
        user_id=user_id,
        agent_id=agent_id,
        workspace_id=workspace_id,
    )
    favorites = await favorite_ids(user_id=user_id, agent_id=agent_id)
    track = _payload_to_track(
        payload,
        is_favorite=payload.id in favorites,
        played_by_agent=True,
    )
    await db.execute_raw(
        """
        INSERT INTO music_playbacks (
            id, user_id, agent_id, workspace_id, track_external_id, title, artist,
            album, library, audio_url, duration_sec, cover_key, accent_a,
            accent_b, source, metadata, position_seconds, is_playing
        )
        VALUES (
            $1, $2, $3, $4, $5, $6, $7,
            $8, $9, $10, $11, $12, $13,
            $14, $15, $16::jsonb, $17, $18
        )
        ON CONFLICT (user_id, agent_id)
        DO UPDATE SET
            workspace_id = EXCLUDED.workspace_id,
            track_external_id = EXCLUDED.track_external_id,
            title = EXCLUDED.title,
            artist = EXCLUDED.artist,
            album = EXCLUDED.album,
            library = EXCLUDED.library,
            audio_url = EXCLUDED.audio_url,
            duration_sec = EXCLUDED.duration_sec,
            cover_key = EXCLUDED.cover_key,
            accent_a = EXCLUDED.accent_a,
            accent_b = EXCLUDED.accent_b,
            source = EXCLUDED.source,
            metadata = EXCLUDED.metadata,
            position_seconds = EXCLUDED.position_seconds,
            is_playing = EXCLUDED.is_playing,
            updated_at = now()
        """,
        str(uuid.uuid4()),
        user_id,
        agent_id,
        workspace_id,
        track.id,
        track.title,
        track.artist,
        track.album,
        track.library,
        track.url,
        track.duration_sec,
        track.cover_key,
        track.accent_a,
        track.accent_b,
        track.source,
        json.dumps(track.metadata, ensure_ascii=False),
        position_seconds,
        is_playing,
    )
    return track, position_seconds, is_playing, None


def _mock_track(library: str, *, index: int) -> MusicTrack:
    title, artist, album, duration = _MOCK_TRACKS[index % len(_MOCK_TRACKS)]
    track_id = _track_id(f"mock:{library}:{title}:{artist}")
    visual_index = _visual_index(track_id)
    accent_a, accent_b = _ACCENTS[visual_index % len(_ACCENTS)]
    return MusicTrack(
        id=track_id,
        title=title,
        artist=artist,
        album=album,
        library=library,
        url="",
        duration_sec=duration,
        cover_key=f"music-cover-{(visual_index % 11) + 1:02d}.jpg",
        accent_a=accent_a,
        accent_b=accent_b,
        source="mock",
    )


def _payload_to_track(
    payload: MusicTrackPayload,
    *,
    is_favorite: bool = False,
    played_by_agent: bool = False,
) -> MusicTrack:
    return MusicTrack(
        id=payload.id,
        title=payload.title,
        artist=payload.artist,
        album=payload.album,
        library=payload.library,
        url=payload.url,
        duration_sec=payload.duration_sec,
        cover_key=payload.cover_key,
        accent_a=payload.accent_a,
        accent_b=payload.accent_b,
        source=payload.source,
        is_favorite=is_favorite,
        played_by_agent=played_by_agent,
        metadata=payload.metadata,
    )


def _favorite_row_to_track(row: dict[str, Any]) -> MusicTrack:
    return _stored_row_to_track(row, is_favorite=True)


def _playback_row_to_track(row: dict[str, Any]) -> MusicTrack:
    return _stored_row_to_track(row, played_by_agent=True)


def _stored_row_to_track(
    row: dict[str, Any],
    *,
    is_favorite: bool = False,
    played_by_agent: bool = False,
) -> MusicTrack:
    return MusicTrack(
        id=_clean_text(row.get("track_external_id")),
        title=_clean_text(row.get("title")) or "Untitled Audio",
        artist=_clean_text(row.get("artist")) or "AudioLib",
        album=_clean_text(row.get("album")) or "Curated Library",
        library=_clean_text(row.get("library")) or "audio.focus",
        url=_clean_text(row.get("audio_url")),
        duration_sec=_int(row.get("duration_sec")),
        cover_key=_clean_text(row.get("cover_key")) or "music-cover-01.jpg",
        accent_a=_clean_text(row.get("accent_a")) or "#1f6fff",
        accent_b=_clean_text(row.get("accent_b")) or "#18c6c0",
        source=_clean_text(row.get("source")) or "audiolib",
        is_favorite=is_favorite,
        played_by_agent=played_by_agent,
        metadata=_json_value(row.get("metadata"), {}),
    )


def _track_id(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]


def _visual_index(seed: str) -> int:
    return int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8], 16)


def _library_title(library: str) -> str:
    tail = library.split(".")[-1].replace("_", " ").replace("-", " ").strip()
    return f"AudioLib {tail.title()}" if tail else "AudioLib Library"


def _library_label(library: str) -> str:
    tail = library.split(".")[-1].replace("_", " ").replace("-", " ").strip()
    return tail.title() if tail else library


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _json_value(value: Any, fallback: Any) -> Any:
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


def _row(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return row
    model_dump = getattr(row, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    if hasattr(row, "__dict__"):
        return dict(row.__dict__)
    return {}


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
