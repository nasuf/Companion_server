from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any

import httpx

from app.config import settings
from app.db import db
from app.models.music import MusicCoListeningResponse, MusicLibrary, MusicTrack, MusicTrackPayload

logger = logging.getLogger(__name__)

JAMENDO_TRACKS_PATH = "/tracks/"
JAMENDO_AUDIO_FORMAT = "mp31"
DEFAULT_LIBRARIES = ["focus", "ambient", "sleep"]
DEFAULT_TRACK_LIMIT = 1
AUDIO_CACHE_TTL_SECONDS = 5 * 60
STREAM_TOKEN_TTL_SECONDS = 60 * 30
_TIMEOUT = httpx.Timeout(8.0, connect=4.0)
_audio_cache: dict[tuple[str, str, str, int], tuple[float, MusicTrack]] = {}
_LIBRARY_CATALOG = {
    "focus": ("专注", "Jamendo 纯音乐"),
    "ambient": ("Ambient", "氛围陪伴"),
    "sleep": ("助眠", "轻柔放松"),
    "relax": ("放松", "Relaxing tag"),
    "acoustic": ("原声", "温暖轻听"),
    "piano": ("钢琴", "安静旋律"),
    "pop": ("流行", "轻松随机"),
    "electronic": ("电子", "节奏漂浮"),
}
_LIBRARY_TAG_ALIASES = {
    "focus": "instrumental",
    "sleep": "relaxing",
    "relax": "relaxing",
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
    return bool(settings.jamendo_client_id.strip())


def missing_config() -> list[str]:
    return [] if api_enabled() else ["JAMENDO_CLIENT_ID"]


def default_libraries() -> list[str]:
    raw = settings.jamendo_default_libraries.strip()
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


async def ensure_conversation_owner(
    *,
    user_id: str,
    agent_id: str,
    conversation_id: str,
) -> dict[str, Any]:
    rows = await db.query_raw(
        """
        SELECT id, user_id, agent_id, workspace_id
        FROM conversations
        WHERE id = $1 AND user_id = $2 AND agent_id = $3 AND is_deleted = false
        LIMIT 1
        """,
        conversation_id,
        user_id,
        agent_id,
    )
    if not rows:
        raise ValueError("conversation_not_found")
    return _row(rows[0])


async def list_square_tracks(
    *,
    user_id: str,
    agent_id: str,
    workspace_id: str | None,
    library: str | None,
    limit: int,
    refresh: bool = False,
    exclude_track_ids: set[str | None] | None = None,
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
    excluded_ids = {_clean_text(item) for item in (exclude_track_ids or set())}
    excluded_ids.discard("")
    for index in range(max(1, limit)):
        chosen = libraries[index % len(libraries)]
        track = await fetch_random_track(
            chosen,
            index=index,
            use_cache=not refresh,
            excluded_ids=excluded_ids,
        )
        excluded_ids.add(track.id)
        tracks.append(track.model_copy(update={"is_favorite": track.id in favorites}))
    return tracks


async def fetch_random_track(
    library: str,
    *,
    index: int = 0,
    use_cache: bool = True,
    excluded_ids: set[str] | None = None,
) -> MusicTrack:
    if not api_enabled():
        return _mock_track(library, index=index)
    base_url = _jamendo_base_url()
    client_id = settings.jamendo_client_id.strip()
    cache_key = (_track_id(client_id), base_url, library, index)
    cached = _audio_cache.get(cache_key)
    now = time.monotonic()
    if use_cache and cached and now - cached[0] < AUDIO_CACHE_TTL_SECONDS:
        return cached[1].model_copy()
    excluded_ids = {_clean_text(item) for item in (excluded_ids or set())}
    excluded_ids.discard("")
    refresh_nonce = None if use_cache else f"{time.time_ns()}:{uuid.uuid4().hex}"
    max_attempts = 6 if excluded_ids else 1
    data: dict[str, Any] | None = None
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, trust_env=False) as client:
            for attempt in range(max_attempts):
                offset = _jamendo_offset(
                    library,
                    index=index + attempt,
                    refresh_nonce=f"{refresh_nonce}:{attempt}" if refresh_nonce else None,
                )
                response = await client.get(
                    f"{base_url}{JAMENDO_TRACKS_PATH}",
                    params=_jamendo_track_params(
                        client_id=client_id,
                        library=library,
                        offset=offset,
                    ),
                )
                response.raise_for_status()
                payload = response.json()
                data = _first_jamendo_result(payload)
                if data is None:
                    fallback = await client.get(
                        f"{base_url}{JAMENDO_TRACKS_PATH}",
                        params=_jamendo_track_params(
                            client_id=client_id,
                            library=None,
                            offset=offset,
                        ),
                    )
                    fallback.raise_for_status()
                    data = _first_jamendo_result(fallback.json())
                candidate_id = _clean_text(data.get("id")) if isinstance(data, dict) else ""
                if candidate_id and candidate_id in excluded_ids and attempt < max_attempts - 1:
                    logger.info(
                        "Jamendo refresh skipped excluded track: library=%s track_id=%s offset=%s",
                        library,
                        candidate_id,
                        offset,
                    )
                    continue
                break
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Jamendo request failed, using mock music: %s", exc)
        return _mock_track(library, index=index)
    if not isinstance(data, dict):
        logger.warning("Jamendo returned no playable track, using mock music")
        return _mock_track(library, index=index)
    track_id = _clean_text(data.get("id")) or _track_id(json.dumps(data, sort_keys=True))
    title = _clean_text(data.get("name")) or "Untitled Track"
    artist = _clean_text(data.get("artist_name")) or "Jamendo"
    album = _clean_text(data.get("album_name")) or _library_title(library)
    duration = _int(data.get("duration"))
    audio_url = _jamendo_audio_url(data)
    if not audio_url:
        logger.warning("Jamendo track %s has no playable audio URL, using mock music", track_id)
        return _mock_track(library, index=index)
    visual_index = _visual_index(track_id)
    accent_a, accent_b = _ACCENTS[visual_index % len(_ACCENTS)]
    musicinfo = data.get("musicinfo") if isinstance(data.get("musicinfo"), dict) else {}
    track = MusicTrack(
        id=track_id,
        title=title,
        artist=artist,
        album=album,
        library=library,
        url=stream_url_path(track_id),
        duration_sec=duration,
        cover_key=f"music-cover-{(visual_index % 11) + 1:02d}.jpg",
        accent_a=accent_a,
        accent_b=accent_b,
        source="jamendo",
        metadata={
            "provider": "jamendo",
            "jamendo_id": track_id,
            "jamendo_tag": _jamendo_tag(library),
            "audio": audio_url,
            "audiodownload": _clean_text(data.get("audiodownload")),
            "image": _clean_text(data.get("image") or data.get("album_image")),
            "album_image": _clean_text(data.get("album_image")),
            "lyrics": _clean_text(data.get("lyrics")),
            "vocalinstrumental": _clean_text(musicinfo.get("vocalinstrumental")),
            "lang": _clean_text(musicinfo.get("lang")),
            "license_ccurl": _clean_text(data.get("license_ccurl")),
            "audiodownload_allowed": bool(data.get("audiodownload_allowed")),
            "raw": data,
        },
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


async def resolve_play_url(
    *,
    user_id: str,
    agent_id: str,
    track_id: str,
) -> str:
    await ensure_agent_owner(user_id, agent_id)
    if not api_enabled():
        return ""
    clean_track_id = _clean_text(track_id)
    if not clean_track_id:
        raise ValueError("track_not_found")
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, trust_env=False) as client:
            response = await client.get(
                f"{_jamendo_base_url()}{JAMENDO_TRACKS_PATH}",
                params=_jamendo_track_lookup_params(
                    client_id=settings.jamendo_client_id.strip(),
                    track_id=clean_track_id,
                ),
            )
            response.raise_for_status()
            data = _first_jamendo_result(response.json())
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Jamendo play URL refresh failed: %s", exc)
        return ""
    url = _jamendo_audio_url(data)
    if not url:
        raise ValueError("track_not_found")
    return stream_url_path(clean_track_id)


async def resolve_stream_audio_url(track_id: str) -> str:
    urls = await resolve_stream_audio_urls(track_id)
    return urls[0] if urls else ""


async def resolve_stream_audio_urls(track_id: str) -> list[str]:
    if not api_enabled():
        return []
    clean_track_id = _clean_text(track_id)
    if not clean_track_id:
        return []
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, trust_env=False) as client:
            response = await client.get(
                f"{_jamendo_base_url()}{JAMENDO_TRACKS_PATH}",
                params=_jamendo_track_lookup_params(
                    client_id=settings.jamendo_client_id.strip(),
                    track_id=clean_track_id,
                ),
            )
            response.raise_for_status()
            data = _first_jamendo_result(response.json())
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Jamendo stream URL lookup failed: %s", exc)
        return []
    return _jamendo_stream_urls(data)


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
    conversation_id: str | None = None,
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
    if conversation_id:
        await update_active_co_listening(
            user_id=user_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            payload=payload,
            position_seconds=position_seconds,
            is_playing=is_playing,
        )
    return track, position_seconds, is_playing, None


async def get_active_co_listening(
    *,
    conversation_id: str,
) -> MusicCoListeningResponse | None:
    rows = await db.query_raw(
        """
        SELECT *
        FROM music_co_listening_sessions
        WHERE conversation_id = $1 AND status = 'active'
        LIMIT 1
        """,
        conversation_id,
    )
    if not rows:
        return None
    return _co_listening_row_to_response(_row(rows[0]))


async def start_co_listening(
    *,
    user_id: str,
    agent_id: str,
    conversation_id: str,
    workspace_id: str | None,
    payload: MusicTrackPayload,
    initiated_by: str = "user",
    position_seconds: int = 0,
    is_playing: bool = True,
) -> MusicCoListeningResponse:
    await ensure_agent_owner(user_id, agent_id)
    conversation = await ensure_conversation_owner(
        user_id=user_id,
        agent_id=agent_id,
        conversation_id=conversation_id,
    )
    resolved_workspace_id = await resolve_workspace(
        user_id=user_id,
        agent_id=agent_id,
        workspace_id=workspace_id or _clean_text(conversation.get("workspace_id")),
    )
    track = _payload_to_track(payload, played_by_agent=True)
    await db.execute_raw(
        """
        INSERT INTO music_co_listening_sessions (
            id, user_id, agent_id, workspace_id, conversation_id, status,
            initiated_by, track_external_id, title, artist, album, library,
            audio_url, duration_sec, cover_key, accent_a, accent_b, source,
            metadata, position_seconds, is_playing, ended_reason, ended_at
        )
        VALUES (
            $1, $2, $3, $4, $5, 'active',
            $6, $7, $8, $9, $10, $11,
            $12, $13, $14, $15, $16, $17,
            $18::jsonb, $19, $20, NULL, NULL
        )
        ON CONFLICT (conversation_id)
        DO UPDATE SET
            user_id = EXCLUDED.user_id,
            agent_id = EXCLUDED.agent_id,
            workspace_id = EXCLUDED.workspace_id,
            status = 'active',
            initiated_by = EXCLUDED.initiated_by,
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
            ended_reason = NULL,
            ended_at = NULL,
            updated_at = now()
        """,
        str(uuid.uuid4()),
        user_id,
        agent_id,
        resolved_workspace_id,
        conversation_id,
        initiated_by,
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
    return MusicCoListeningResponse(
        status="active",
        track=track,
        position_seconds=position_seconds,
        is_playing=is_playing,
        initiated_by=initiated_by,
    )


async def update_active_co_listening(
    *,
    user_id: str,
    agent_id: str,
    conversation_id: str,
    payload: MusicTrackPayload,
    position_seconds: int,
    is_playing: bool,
) -> bool:
    await ensure_conversation_owner(
        user_id=user_id,
        agent_id=agent_id,
        conversation_id=conversation_id,
    )
    track = _payload_to_track(payload, played_by_agent=True)
    count = await db.execute_raw(
        """
        UPDATE music_co_listening_sessions
        SET track_external_id = $4,
            title = $5,
            artist = $6,
            album = $7,
            library = $8,
            audio_url = $9,
            duration_sec = $10,
            cover_key = $11,
            accent_a = $12,
            accent_b = $13,
            source = $14,
            metadata = $15::jsonb,
            position_seconds = $16,
            is_playing = $17,
            updated_at = now()
        WHERE conversation_id = $1
          AND user_id = $2
          AND agent_id = $3
          AND status = 'active'
        """,
        conversation_id,
        user_id,
        agent_id,
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
    return bool(count)


async def end_co_listening(
    *,
    user_id: str,
    agent_id: str,
    conversation_id: str,
    reason: str = "user_exit",
) -> MusicCoListeningResponse | None:
    await ensure_conversation_owner(
        user_id=user_id,
        agent_id=agent_id,
        conversation_id=conversation_id,
    )
    current = await get_active_co_listening(conversation_id=conversation_id)
    if current is None:
        return None
    await db.execute_raw(
        """
        UPDATE music_co_listening_sessions
        SET status = 'ended',
            ended_reason = $4,
            ended_at = now(),
            is_playing = false,
            updated_at = now()
        WHERE conversation_id = $1
          AND user_id = $2
          AND agent_id = $3
          AND status = 'active'
        """,
        conversation_id,
        user_id,
        agent_id,
        reason,
    )
    return current.model_copy(
        update={"status": "ended", "is_playing": False, "ended_reason": reason}
    )


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


def _jamendo_base_url() -> str:
    return settings.jamendo_base_url.strip().rstrip("/") or "https://api.jamendo.com/v3.0"


def _jamendo_track_params(
    *,
    client_id: str,
    library: str | None,
    offset: int,
) -> dict[str, str | int]:
    params: dict[str, str | int] = {
        "client_id": client_id,
        "format": "json",
        "limit": 1,
        "offset": max(0, offset),
        "include": "musicinfo+lyrics",
        "audioformat": JAMENDO_AUDIO_FORMAT,
        "audiodlformat": JAMENDO_AUDIO_FORMAT,
        "order": "popularity_total",
    }
    if library:
        params["tags"] = _jamendo_tag(library)
    return params


def _jamendo_track_lookup_params(
    *,
    client_id: str,
    track_id: str,
) -> dict[str, str | int]:
    return {
        "client_id": client_id,
        "format": "json",
        "id": track_id,
        "include": "musicinfo+lyrics",
        "audioformat": JAMENDO_AUDIO_FORMAT,
        "audiodlformat": JAMENDO_AUDIO_FORMAT,
    }


def _jamendo_tag(library: str) -> str:
    return _LIBRARY_TAG_ALIASES.get(library, library)


def _first_jamendo_result(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return None
    first = results[0]
    return first if isinstance(first, dict) else None


def _jamendo_offset(
    library: str,
    *,
    index: int,
    refresh_nonce: str | None = None,
) -> int:
    # Jamendo has no random endpoint; rotating the offset gives refreshes variety
    # while cache keeps ordinary page reloads cheap.
    bucket = refresh_nonce or str(int(time.time() // 300))
    seed = f"{library}:{index}:{bucket}"
    return _visual_index(seed) % 200


def _jamendo_audio_url(data: dict[str, Any] | None) -> str:
    if not isinstance(data, dict):
        return ""
    return _clean_text(data.get("audio"))


def _jamendo_stream_urls(data: dict[str, Any] | None) -> list[str]:
    if not isinstance(data, dict):
        return []
    candidates = [
        _clean_text(data.get("audiodownload")),
        _clean_text(data.get("audio")),
    ]
    urls: list[str] = []
    seen: set[str] = set()
    for url in candidates:
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def stream_url_path(track_id: str) -> str:
    clean_track_id = _clean_text(track_id)
    if not clean_track_id:
        return ""
    return f"/music/tracks/{clean_track_id}/stream.mp3?token={stream_token(clean_track_id)}"


def stream_token(track_id: str, *, now: int | None = None) -> str:
    expires_at = (now or int(time.time())) + STREAM_TOKEN_TTL_SECONDS
    message = f"{track_id}:{expires_at}"
    signature = hmac.new(
        _stream_secret().encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{expires_at}.{signature}"


def validate_stream_token(track_id: str, token: str) -> bool:
    try:
        expires_raw, signature = token.split(".", 1)
        expires_at = int(expires_raw)
    except (ValueError, AttributeError):
        return False
    if expires_at < int(time.time()):
        return False
    expected = stream_token(track_id, now=expires_at - STREAM_TOKEN_TTL_SECONDS)
    return hmac.compare_digest(token, expected)


def _stream_secret() -> str:
    return settings.jwt_secret.strip() or settings.jamendo_client_id.strip() or "dev-music-stream"


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


def _co_listening_row_to_response(row: dict[str, Any]) -> MusicCoListeningResponse:
    return MusicCoListeningResponse(
        status=_clean_text(row.get("status")) or "ended",
        track=_stored_row_to_track(row, played_by_agent=True),
        position_seconds=_int(row.get("position_seconds")),
        is_playing=bool(row.get("is_playing")),
        initiated_by=_clean_text(row.get("initiated_by")) or None,
        ended_reason=_clean_text(row.get("ended_reason")) or None,
        updated_at=_iso(row.get("updated_at")),
    )


def _stored_row_to_track(
    row: dict[str, Any],
    *,
    is_favorite: bool = False,
    played_by_agent: bool = False,
) -> MusicTrack:
    return MusicTrack(
        id=_clean_text(row.get("track_external_id")),
        title=_clean_text(row.get("title")) or "Untitled Track",
        artist=_clean_text(row.get("artist")) or "Jamendo",
        album=_clean_text(row.get("album")) or "Jamendo Library",
        library=_clean_text(row.get("library")) or "focus",
        url=_clean_text(row.get("audio_url")),
        duration_sec=_int(row.get("duration_sec")),
        cover_key=_clean_text(row.get("cover_key")) or "music-cover-01.jpg",
        accent_a=_clean_text(row.get("accent_a")) or "#1f6fff",
        accent_b=_clean_text(row.get("accent_b")) or "#18c6c0",
        source=_clean_text(row.get("source")) or "jamendo",
        is_favorite=is_favorite,
        played_by_agent=played_by_agent,
        metadata=_json_value(row.get("metadata"), {}),
    )


def _track_id(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]


def _visual_index(seed: str) -> int:
    return int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8], 16)


def _library_title(library: str) -> str:
    catalog = _LIBRARY_CATALOG.get(library)
    if catalog:
        return f"Jamendo {catalog[0]}"
    tail = library.split(".")[-1].replace("_", " ").replace("-", " ").strip()
    return f"Jamendo {tail.title()}" if tail else "Jamendo Library"


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
