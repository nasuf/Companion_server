from __future__ import annotations

import pytest
from starlette.requests import Request
from unittest.mock import AsyncMock

from app.api.public import music as music_api
from app.models.music import MusicCoListeningEndRequest, MusicCoListeningResponse, MusicPlaybackRequest, MusicTrack
from app.models.music import MusicTrackPayload
from app.services import music
from app.services import music_status


class _FakeDb:
    def __init__(self, rows_by_query: list[list[dict]] | None = None):
        self.rows_by_query = rows_by_query or []
        self.queries: list[tuple[str, tuple]] = []
        self.execs: list[tuple[str, tuple]] = []

    async def query_raw(self, query: str, *args):
        self.queries.append((query, args))
        if self.rows_by_query:
            return self.rows_by_query.pop(0)
        return []

    async def execute_raw(self, query: str, *args):
        self.execs.append((query, args))
        return 1


@pytest.mark.asyncio
async def test_list_square_tracks_marks_existing_favorites(monkeypatch):
    fake_db = _FakeDb(
        [
            [{"id": "agent-1", "name": "小芜", "user_id": "user-1"}],
            [{"track_external_id": "fav-track"}],
        ]
    )
    monkeypatch.setattr(music, "db", fake_db)

    async def fake_fetch_random_track(
        library: str,
        *,
        index: int = 0,
        use_cache: bool = True,
        excluded_ids: set[str] | None = None,
    ):
        return music.MusicTrack(
            id="fav-track" if index == 0 else f"track-{index}",
            title=f"Track {index}",
            library=library,
        )

    monkeypatch.setattr(music, "fetch_random_track", fake_fetch_random_track)

    tracks = await music.list_square_tracks(
        user_id="user-1",
        agent_id="agent-1",
        workspace_id=None,
        library="focus",
        limit=2,
    )

    assert [track.is_favorite for track in tracks] == [True, False]


@pytest.mark.asyncio
async def test_list_square_tracks_refresh_bypasses_audio_cache(monkeypatch):
    fake_db = _FakeDb(
        [
            [{"id": "agent-1", "name": "小芜", "user_id": "user-1"}],
            [],
        ]
    )
    monkeypatch.setattr(music, "db", fake_db)
    use_cache_values = []

    async def fake_fetch_random_track(
        library: str,
        *,
        index: int = 0,
        use_cache: bool = True,
        excluded_ids: set[str] | None = None,
    ):
        use_cache_values.append(use_cache)
        return music.MusicTrack(
            id="track-1",
            title="Track 1",
            library=library,
        )

    monkeypatch.setattr(music, "fetch_random_track", fake_fetch_random_track)

    await music.list_square_tracks(
        user_id="user-1",
        agent_id="agent-1",
        workspace_id=None,
        library="focus",
        limit=1,
        refresh=True,
    )

    assert use_cache_values == [False]


@pytest.mark.asyncio
async def test_list_square_tracks_passes_current_track_exclusion(monkeypatch):
    fake_db = _FakeDb(
        [
            [{"id": "agent-1", "name": "小芜", "user_id": "user-1"}],
            [],
        ]
    )
    monkeypatch.setattr(music, "db", fake_db)
    exclusions = []

    async def fake_fetch_random_track(
        library: str,
        *,
        index: int = 0,
        use_cache: bool = True,
        excluded_ids: set[str] | None = None,
    ):
        exclusions.append(set(excluded_ids or set()))
        return music.MusicTrack(id="next-track", title="Next Track", library=library)

    monkeypatch.setattr(music, "fetch_random_track", fake_fetch_random_track)

    tracks = await music.list_square_tracks(
        user_id="user-1",
        agent_id="agent-1",
        workspace_id=None,
        library="focus",
        limit=1,
        refresh=True,
        exclude_track_ids={"current-track"},
    )

    assert tracks[0].id == "next-track"
    assert exclusions == [{"current-track"}]


@pytest.mark.asyncio
async def test_add_favorite_upserts_track(monkeypatch):
    fake_db = _FakeDb([[{"id": "agent-1", "name": "小芜", "user_id": "user-1"}]])
    monkeypatch.setattr(music, "db", fake_db)

    track = await music.add_favorite(
        user_id="user-1",
        agent_id="agent-1",
        workspace_id=None,
        payload=MusicTrackPayload(
            id="track-1",
            title="Deep Space Drift",
            artist="Jamendo Artist",
            album="Jamendo Focus",
            library="focus",
            url="https://cdn.example.test/audio.mp3",
            duration_sec=240,
        ),
    )

    assert track.is_favorite
    assert track.id == "track-1"
    assert fake_db.execs
    assert "ON CONFLICT" in fake_db.execs[0][0]


@pytest.mark.asyncio
async def test_now_playing_is_empty_until_client_starts_random_track(monkeypatch):
    fake_db = _FakeDb(
        [
            [{"id": "agent-1", "name": "小芜", "user_id": "user-1"}],
            [],
            [],
        ]
    )
    monkeypatch.setattr(music, "db", fake_db)

    track, position_seconds, is_playing, updated_at = await music.get_now_playing(
        user_id="user-1",
        agent_id="agent-1",
    )

    assert track is None
    assert position_seconds == 0
    assert not is_playing
    assert updated_at is None


@pytest.mark.asyncio
async def test_fetch_random_track_uses_short_ttl_cache(monkeypatch):
    calls = []

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [
                    {
                        "id": "12345",
                        "name": "Deep Space Drift",
                        "artist_name": "Jamendo Artist",
                        "album_name": "Jamendo Focus",
                        "duration": 240,
                        "audio": "https://prod-1.storage.jamendo.com/?trackid=12345&format=mp32&from=test",
                        "audiodownload": "https://prod-1.storage.jamendo.com/download/track/12345/mp31/",
                        "image": "https://usercontent.jamendo.com/cover.jpg",
                        "album_image": "https://usercontent.jamendo.com/album.jpg",
                        "lyrics": "first line\nsecond line",
                        "musicinfo": {
                            "vocalinstrumental": "vocal",
                            "lang": "en",
                        },
                    }
                ]
            }

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, *args, **kwargs):
            calls.append((args, kwargs))
            return _FakeResponse()

    music._audio_cache.clear()
    monkeypatch.setattr(music.settings, "jamendo_client_id", "jam_test_client")
    monkeypatch.setattr(music.settings, "jamendo_base_url", "https://api.jamendo.com/v3.0")
    monkeypatch.setattr(music.httpx, "AsyncClient", _FakeAsyncClient)

    first = await music.fetch_random_track("focus", index=0)
    second = await music.fetch_random_track("focus", index=0)

    assert first.id == second.id
    assert first.url.startswith("/music/tracks/12345/stream.mp3?token=")
    assert music.validate_stream_token("12345", first.url.split("token=", 1)[1])
    assert first.source == "jamendo"
    assert first.metadata["audio"] == "https://prod-1.storage.jamendo.com/?trackid=12345&format=mp32&from=test"
    assert first.metadata["audiodownload"] == "https://prod-1.storage.jamendo.com/download/track/12345/mp31/"
    assert first.metadata["image"] == "https://usercontent.jamendo.com/cover.jpg"
    assert first.metadata["lyrics"] == "first line\nsecond line"
    assert first.metadata["vocalinstrumental"] == "vocal"
    assert first.metadata["lang"] == "en"
    assert calls[0][1]["params"]["audioformat"] == "mp31"
    assert calls[0][1]["params"]["audiodlformat"] == "mp31"
    assert calls[0][1]["params"]["include"] == "musicinfo+lyrics"
    assert calls[0][1]["params"]["tags"] == "instrumental"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_fetch_random_track_refresh_skips_excluded_track(monkeypatch):
    calls = []
    responses = [
        {
            "id": "current-track",
            "name": "Current Track",
            "artist_name": "Jamendo Artist",
            "duration": 240,
            "audio": "https://prod-1.storage.jamendo.com/?trackid=current-track&format=mp31",
        },
        {
            "id": "next-track",
            "name": "Next Track",
            "artist_name": "Jamendo Artist",
            "duration": 242,
            "audio": "https://prod-1.storage.jamendo.com/?trackid=next-track&format=mp31",
        },
    ]

    class _FakeResponse:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            return None

        def json(self):
            return {"results": [self._data]}

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, *args, **kwargs):
            calls.append((args, kwargs))
            return _FakeResponse(responses[len(calls) - 1])

    monkeypatch.setattr(music.settings, "jamendo_client_id", "jam_test_client")
    monkeypatch.setattr(music.settings, "jamendo_base_url", "https://api.jamendo.com/v3.0")
    monkeypatch.setattr(music.httpx, "AsyncClient", _FakeAsyncClient)

    track = await music.fetch_random_track(
        "focus",
        index=0,
        use_cache=False,
        excluded_ids={"current-track"},
    )

    assert track.id == "next-track"
    assert len(calls) == 2
    assert calls[0][1]["params"]["offset"] != calls[1][1]["params"]["offset"]


def test_mock_track_cover_uses_track_identity_not_list_index():
    focus = music._mock_track("focus", index=0)
    relax = music._mock_track("relax", index=0)

    assert focus.cover_key != relax.cover_key


@pytest.mark.asyncio
async def test_resolve_play_url_refreshes_jamendo_audio_url(monkeypatch):
    fake_db = _FakeDb([[{"id": "agent-1", "name": "小芜", "user_id": "user-1"}]])

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [
                    {
                        "id": "12345",
                        "audio": "https://prod-1.storage.jamendo.com/?trackid=12345&format=mp32&from=refresh",
                    }
                ]
            }

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, *args, **kwargs):
            return _FakeResponse()

    monkeypatch.setattr(music, "db", fake_db)
    monkeypatch.setattr(music.settings, "jamendo_client_id", "jam_test_client")
    monkeypatch.setattr(music.settings, "jamendo_base_url", "https://api.jamendo.com/v3.0")
    monkeypatch.setattr(music.httpx, "AsyncClient", _FakeAsyncClient)

    url = await music.resolve_play_url(
        user_id="user-1",
        agent_id="agent-1",
        track_id="12345",
    )

    assert url.startswith("/music/tracks/12345/stream.mp3?token=")
    assert music.validate_stream_token("12345", url.split("token=", 1)[1])


@pytest.mark.asyncio
async def test_resolve_play_url_returns_fresh_proxy_token_when_jamendo_lookup_fails(monkeypatch):
    fake_db = _FakeDb([[{"id": "agent-1", "name": "小芜", "user_id": "user-1"}]])

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, *args, **kwargs):
            raise music.httpx.ConnectTimeout("timeout")

    monkeypatch.setattr(music, "db", fake_db)
    monkeypatch.setattr(music.settings, "jamendo_client_id", "jam_test_client")
    monkeypatch.setattr(music.settings, "jamendo_base_url", "https://api.jamendo.com/v3.0")
    monkeypatch.setattr(music.httpx, "AsyncClient", _FakeAsyncClient)

    url = await music.resolve_play_url(
        user_id="user-1",
        agent_id="agent-1",
        track_id="12345",
    )

    assert url.startswith("/music/tracks/12345/stream.mp3?token=")
    assert music.validate_stream_token("12345", url.split("token=", 1)[1])


@pytest.mark.asyncio
async def test_resolve_stream_audio_urls_prefers_download_candidate(monkeypatch):
    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [
                    {
                        "id": "12345",
                        "audio": "https://prod-1.storage.jamendo.com/?trackid=12345&format=mp31&from=stream",
                        "audiodownload": "https://prod-1.storage.jamendo.com/download/track/12345/mp31/",
                    }
                ]
            }

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, *args, **kwargs):
            return _FakeResponse()

    monkeypatch.setattr(music.settings, "jamendo_client_id", "jam_test_client")
    monkeypatch.setattr(music.settings, "jamendo_base_url", "https://api.jamendo.com/v3.0")
    monkeypatch.setattr(music.httpx, "AsyncClient", _FakeAsyncClient)

    urls = await music.resolve_stream_audio_urls("12345")

    assert urls == [
        "https://prod-1.storage.jamendo.com/download/track/12345/mp31/",
        "https://prod-1.storage.jamendo.com/?trackid=12345&format=mp31&from=stream",
    ]


@pytest.mark.asyncio
async def test_stream_music_track_adds_default_range_and_streams_audio(monkeypatch):
    attempts = []

    class _FakeUpstream:
        status_code = 206
        headers = {
            "content-type": "audio/mpeg",
            "accept-ranges": "bytes",
            "content-range": "bytes 0-4/5",
            "content-length": "5",
        }

        async def aiter_bytes(self):
            yield b"audio"

    class _FakeStreamContext:
        def __init__(self, url, headers):
            self.url = url
            self.headers = headers

        async def __aenter__(self):
            attempts.append((self.url, self.headers))
            return _FakeUpstream()

        async def __aexit__(self, exc_type, exc, tb):
            return None

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        def stream(self, method, url, *, headers):
            return _FakeStreamContext(url, headers)

        async def aclose(self):
            return None

    async def fake_resolve_stream_audio_urls(track_id: str):
        return ["https://prod-1.storage.jamendo.com/download/track/12345/mp31/"]

    monkeypatch.setattr(music, "resolve_stream_audio_urls", fake_resolve_stream_audio_urls)
    monkeypatch.setattr(music_api.httpx, "AsyncClient", _FakeAsyncClient)
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/music/tracks/12345/stream.mp3",
            "headers": [],
            "query_string": b"",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
            "scheme": "http",
        }
    )

    response = await music_api.stream_music_track(
        "12345",
        music.stream_token("12345"),
        request,
    )
    body = b"".join([chunk async for chunk in response.body_iterator])
    if response.background is not None:
        await response.background()

    assert response.status_code == 206
    assert response.headers["content-type"].startswith("audio/mpeg")
    assert attempts[0][1]["Range"] == "bytes=0-"
    assert body == b"audio"


@pytest.mark.asyncio
async def test_start_co_listening_upserts_conversation_state(monkeypatch):
    fake_db = _FakeDb(
        [
            [{"id": "agent-1", "name": "小芜", "user_id": "user-1"}],
            [{"id": "conv-1", "user_id": "user-1", "agent_id": "agent-1", "workspace_id": None}],
        ]
    )
    monkeypatch.setattr(music, "db", fake_db)

    result = await music.start_co_listening(
        user_id="user-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        workspace_id=None,
        payload=MusicTrackPayload(id="track-1", title="Quiet Realm"),
    )

    assert result.status == "active"
    assert result.track and result.track.id == "track-1"
    assert fake_db.execs
    assert "music_co_listening_sessions" in fake_db.execs[0][0]
    assert "ON CONFLICT (conversation_id)" in fake_db.execs[0][0]


@pytest.mark.asyncio
async def test_ensure_idle_auto_listening_starts_track(monkeypatch):
    fake_db = _FakeDb(
        [
            [],
            [],
            [{"id": "agent-1", "name": "小芜", "user_id": "user-1"}],
            [{"id": "conv-1", "user_id": "user-1", "agent_id": "agent-1", "workspace_id": None}],
            [{"id": "agent-1", "name": "小芜", "user_id": "user-1"}],
            [],
        ]
    )
    monkeypatch.setattr(music, "db", fake_db)

    async def fake_fetch_random_track(library: str, *, index: int = 0, use_cache: bool = True, excluded_ids=None):
        return music.MusicTrack(
            id="auto-track",
            title="Auto Track",
            artist="Jamendo Artist",
            library=library,
            url="/music/tracks/auto-track/stream.mp3?token=test",
            duration_sec=180,
        )

    monkeypatch.setattr(music, "fetch_random_track", fake_fetch_random_track)

    result = await music.ensure_idle_auto_listening(
        user_id="user-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        workspace_id=None,
        schedule_status="idle",
    )

    assert result is not None
    assert result.status == "active"
    assert result.initiated_by == "agent_auto"
    assert result.track and result.track.id == "auto-track"
    assert len(fake_db.execs) == 2
    assert fake_db.execs[0][1][5] == "agent_auto"
    assert "INSERT INTO music_playbacks" in fake_db.execs[1][0]


@pytest.mark.asyncio
async def test_ensure_idle_auto_listening_respects_user_exit_cooldown(monkeypatch):
    fake_db = _FakeDb([[], [{"cooldown": 1}]])
    monkeypatch.setattr(music, "db", fake_db)

    async def fake_fetch_random_track(*args, **kwargs):
        raise AssertionError("should not fetch during user-exit cooldown")

    monkeypatch.setattr(music, "fetch_random_track", fake_fetch_random_track)

    result = await music.ensure_idle_auto_listening(
        user_id="user-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        workspace_id=None,
        schedule_status="idle",
    )

    assert result is None
    assert not fake_db.execs


@pytest.mark.asyncio
async def test_ensure_idle_auto_listening_ends_agent_auto_when_busy(monkeypatch):
    active_row = {
        "status": "active",
        "track_external_id": "auto-track",
        "title": "Auto Track",
        "artist": "Jamendo Artist",
        "album": "Jamendo Focus",
        "library": "focus",
        "audio_url": "",
        "duration_sec": 180,
        "cover_key": "music-cover-01.jpg",
        "accent_a": "#1f6fff",
        "accent_b": "#18c6c0",
        "source": "jamendo",
        "metadata": {},
        "position_seconds": 0,
        "is_playing": True,
        "initiated_by": "agent_auto",
    }
    fake_db = _FakeDb(
        [
            [active_row],
            [{"id": "conv-1", "user_id": "user-1", "agent_id": "agent-1", "workspace_id": None}],
            [active_row],
        ]
    )
    monkeypatch.setattr(music, "db", fake_db)

    result = await music.ensure_idle_auto_listening(
        user_id="user-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        workspace_id=None,
        schedule_status="busy",
    )

    assert result is None
    assert "UPDATE music_co_listening_sessions" in fake_db.execs[0][0]
    assert fake_db.execs[0][1][-1] == "ai_busy"


@pytest.mark.asyncio
async def test_update_active_co_listening_updates_current_track(monkeypatch):
    fake_db = _FakeDb(
        [[{"id": "conv-1", "user_id": "user-1", "agent_id": "agent-1", "workspace_id": None}]]
    )
    monkeypatch.setattr(music, "db", fake_db)

    updated = await music.update_active_co_listening(
        user_id="user-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        payload=MusicTrackPayload(id="track-2", title="New Track"),
        position_seconds=42,
        is_playing=True,
    )

    assert updated
    assert "UPDATE music_co_listening_sessions" in fake_db.execs[0][0]
    assert fake_db.execs[0][1][3] == "track-2"
    assert fake_db.execs[0][1][-2:] == (42, True)


@pytest.mark.asyncio
async def test_update_active_co_listening_does_not_promote_pending_agent(monkeypatch):
    fake_db = _FakeDb(
        [[{"id": "conv-1", "user_id": "user-1", "agent_id": "agent-1", "workspace_id": None}]]
    )
    monkeypatch.setattr(music, "db", fake_db)

    updated = await music.update_active_co_listening(
        user_id="user-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        payload=MusicTrackPayload(id="track-2", title="New Track"),
        position_seconds=42,
        is_playing=True,
    )

    assert updated
    query = fake_db.execs[0][0]
    assert "status <> 'pending_agent'" in query
    assert "WHEN $17 THEN 'active'" not in query


@pytest.mark.asyncio
async def test_get_active_co_listening_excludes_user_pending(monkeypatch):
    fake_db = _FakeDb()
    monkeypatch.setattr(music, "db", fake_db)

    result = await music.get_active_co_listening(conversation_id="conv-1")

    assert result is None
    query = fake_db.queries[0][0]
    assert "status = 'active'" in query
    assert "initiated_by <> 'user_pending'" in query


@pytest.mark.asyncio
async def test_end_co_listening_marks_session_ended(monkeypatch):
    fake_db = _FakeDb(
        [
            [{"id": "conv-1", "user_id": "user-1", "agent_id": "agent-1", "workspace_id": None}],
            [{
                "status": "active",
                "track_external_id": "track-1",
                "title": "Quiet Realm",
                "artist": "Jamendo Artist",
                "album": "Jamendo Focus",
                "library": "focus",
                "audio_url": "",
                "duration_sec": 240,
                "cover_key": "music-cover-01.jpg",
                "accent_a": "#1f6fff",
                "accent_b": "#18c6c0",
                "source": "jamendo",
                "metadata": {},
                "position_seconds": 12,
                "is_playing": True,
                "initiated_by": "user",
            }],
        ]
    )
    monkeypatch.setattr(music, "db", fake_db)

    ended = await music.end_co_listening(
        user_id="user-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        reason="user_exit",
    )

    assert ended is not None
    assert ended.status == "ended"
    assert ended.ended_reason == "user_exit"
    assert not ended.is_playing
    assert "ended_at = now()" in fake_db.execs[0][0]


@pytest.mark.asyncio
async def test_reconcile_pending_music_joins_when_agent_becomes_idle(monkeypatch):
    track = MusicTrack(id="track-1", title="Quiet Realm", artist="Jamendo Artist")
    pending = MusicCoListeningResponse(
        status="pending_agent",
        track=track,
        position_seconds=18,
        is_playing=True,
        initiated_by="user",
    )
    start = AsyncMock(return_value=pending.model_copy(update={"status": "active"}))
    emit_reply = AsyncMock(return_value="assistant-msg")
    persist_status = AsyncMock(return_value="status-msg")
    monkeypatch.setattr(music, "get_open_co_listening", AsyncMock(return_value=pending))
    monkeypatch.setattr(music, "start_co_listening", start)
    monkeypatch.setattr(music_status, "_emit_rendered_reply", emit_reply)
    monkeypatch.setattr(music_status, "persist_and_emit_music_status", persist_status)

    result = await music_status.reconcile_co_listening_for_status(
        user_id="user-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        workspace_id="ws-1",
        status_code="idle",
        activity="自由时间",
        ai_name="小芜",
    )

    assert result is not None
    start.assert_awaited_once()
    assert start.await_args.kwargs["status"] == "active"
    assert start.await_args.kwargs["position_seconds"] == 18
    emit_reply.assert_awaited_once()
    assert emit_reply.await_args.kwargs["prompt_key"] == "music.agent_join_after_busy"
    persist_status.assert_awaited_once()
    assert persist_status.await_args.kwargs["actor"] == "agent"
    assert persist_status.await_args.kwargs["actor_name"] == "小芜"


@pytest.mark.asyncio
async def test_reconcile_user_pending_active_music_joins_when_agent_becomes_idle(monkeypatch):
    pending = MusicCoListeningResponse(
        status="active",
        track=MusicTrack(id="track-1", title="Quiet Realm"),
        position_seconds=12,
        is_playing=True,
        initiated_by="user_pending",
    )
    joined = pending.model_copy(update={"initiated_by": "user"})
    start = AsyncMock(return_value=joined)
    emit_reply = AsyncMock(return_value="assistant-msg")
    persist_status = AsyncMock(return_value="status-msg")
    monkeypatch.setattr(music, "get_open_co_listening", AsyncMock(return_value=pending))
    monkeypatch.setattr(music, "start_co_listening", start)
    monkeypatch.setattr(music_status, "_emit_rendered_reply", emit_reply)
    monkeypatch.setattr(music_status, "persist_and_emit_music_status", persist_status)

    result = await music_status.reconcile_co_listening_for_status(
        user_id="user-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        workspace_id="ws-1",
        status_code="idle",
        activity="自由时间",
        ai_name="小青",
    )

    assert result == joined
    start.assert_awaited_once()
    assert start.await_args.kwargs["initiated_by"] == "user"
    assert start.await_args.kwargs["status"] == "active"
    assert emit_reply.await_args.kwargs["prompt_key"] == "music.agent_join_after_busy"
    assert persist_status.await_args.kwargs["actor"] == "agent"
    assert persist_status.await_args.kwargs["actor_name"] == "小青"


@pytest.mark.asyncio
async def test_reconcile_legacy_active_user_without_agent_status_joins_when_idle(monkeypatch):
    pending = MusicCoListeningResponse(
        status="active",
        track=MusicTrack(id="track-1", title="Quiet Realm"),
        position_seconds=12,
        is_playing=True,
        initiated_by="user",
    )
    joined = pending.model_copy(update={"initiated_by": "user"})
    start = AsyncMock(return_value=joined)
    emit_reply = AsyncMock(return_value="assistant-msg")
    persist_status = AsyncMock(return_value="status-msg")
    monkeypatch.setattr(music, "get_open_co_listening", AsyncMock(return_value=pending))
    monkeypatch.setattr(music, "start_co_listening", start)
    monkeypatch.setattr(music_status, "_agent_join_status_exists", AsyncMock(return_value=False))
    monkeypatch.setattr(music_status, "_emit_rendered_reply", emit_reply)
    monkeypatch.setattr(music_status, "persist_and_emit_music_status", persist_status)

    result = await music_status.reconcile_co_listening_for_status(
        user_id="user-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        workspace_id="ws-1",
        status_code="idle",
        activity="自由时间",
        ai_name="小青",
    )

    assert result == joined
    start.assert_awaited_once()
    emit_reply.assert_awaited_once()
    assert emit_reply.await_args.kwargs["prompt_key"] == "music.agent_join_after_busy"
    assert persist_status.await_args.kwargs["actor"] == "agent"


@pytest.mark.asyncio
async def test_music_status_text_uses_join_and_exit_labels(monkeypatch):
    persisted = AsyncMock(return_value="status-msg")
    send_event = AsyncMock()
    monkeypatch.setattr(music_status, "_persist_assistant_message", persisted)
    monkeypatch.setattr(music_status.manager, "send_event", send_event)

    await music_status.persist_and_emit_music_status(
        conversation_id="conv-1",
        status="started",
        track=MusicTrack(id="track-1", title="Quiet Realm"),
        actor="agent",
        actor_name="小芜",
    )
    await music_status.persist_and_emit_music_status(
        conversation_id="conv-1",
        status="ended",
        track=MusicTrack(id="track-1", title="Quiet Realm"),
        actor="user",
    )

    assert persisted.await_args_list[0].args[1] == "小芜已加入共听"
    assert persisted.await_args_list[1].args[1] == "你已退出共听"
    assert send_event.await_args_list[0].args[2]["text"] == "小芜已加入共听"
    assert send_event.await_args_list[1].args[2]["text"] == "你已退出共听"


@pytest.mark.asyncio
async def test_pause_timeout_active_music_moves_to_agent_waiting(monkeypatch):
    active = MusicCoListeningResponse(
        status="active",
        track=MusicTrack(id="track-1", title="Quiet Realm"),
        is_playing=False,
        initiated_by="user",
    )
    waiting = active.model_copy(update={
        "status": "agent_waiting_user",
        "ended_reason": "user_pause_timeout",
    })
    scheduled = []

    def _fake_fire_background(coro):
        scheduled.append(coro)
        coro.close()
        return None

    monkeypatch.setattr(music, "get_open_co_listening", AsyncMock(return_value=active))
    monkeypatch.setattr(
        music,
        "move_paused_active_co_listening_to_agent_waiting_if_stale",
        AsyncMock(return_value=waiting),
    )
    monkeypatch.setattr(music_status, "_render_exit_reply", AsyncMock(return_value="我还在等你回来一起听。"))
    monkeypatch.setattr(music_status, "_persist_assistant_message", AsyncMock(return_value="assistant-msg"))
    monkeypatch.setattr(music_status.manager, "send_event", AsyncMock())
    monkeypatch.setattr(music_status, "persist_and_emit_music_status", AsyncMock(return_value="status-msg"))
    monkeypatch.setattr(music_status, "fire_background", _fake_fire_background)

    await music_status.end_if_paused_after_timeout(
        user_id="user-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        seconds=0,
    )

    music.move_paused_active_co_listening_to_agent_waiting_if_stale.assert_awaited_once()
    assert music.move_paused_active_co_listening_to_agent_waiting_if_stale.await_args.kwargs["reason"] == "user_pause_timeout"
    music_status._render_exit_reply.assert_awaited_once()
    assert music_status._render_exit_reply.await_args.args[0] == "music.user_pause_exit"
    music_status.persist_and_emit_music_status.assert_awaited_once()
    assert music_status.persist_and_emit_music_status.await_args.kwargs["actor"] == "user"
    assert music_status.persist_and_emit_music_status.await_args.kwargs["reason"] == "user_pause_timeout"
    assert len(scheduled) == 1


@pytest.mark.asyncio
async def test_agent_waiting_timeout_exits_agent(monkeypatch):
    waiting = MusicCoListeningResponse(
        status="agent_waiting_user",
        track=MusicTrack(id="track-1", title="Quiet Realm"),
        is_playing=False,
        initiated_by="user",
    )
    ended = waiting.model_copy(update={
        "status": "ended",
        "ended_reason": "user_absent_timeout",
    })
    monkeypatch.setattr(music, "end_agent_waiting_if_stale", AsyncMock(return_value=ended))
    monkeypatch.setattr(music_status, "_render_exit_reply", AsyncMock(return_value="那我们下次再一起听。"))
    monkeypatch.setattr(music_status, "_persist_assistant_message", AsyncMock(return_value="assistant-msg"))
    monkeypatch.setattr(music_status.manager, "send_event", AsyncMock())
    monkeypatch.setattr(music_status, "_resolve_agent_name", AsyncMock(return_value="小芜"))
    monkeypatch.setattr(music_status, "persist_and_emit_music_status", AsyncMock(return_value="status-msg"))

    await music_status.end_agent_waiting_after_timeout(
        user_id="user-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        seconds=0,
    )

    music.end_agent_waiting_if_stale.assert_awaited_once()
    assert music.end_agent_waiting_if_stale.await_args.kwargs["reason"] == "user_absent_timeout"
    music_status._render_exit_reply.assert_awaited_once()
    assert music_status._render_exit_reply.await_args.args[0] == "music.user_absent_exit"
    music_status.persist_and_emit_music_status.assert_awaited_once()
    assert music_status.persist_and_emit_music_status.await_args.kwargs["actor"] == "agent"
    assert music_status.persist_and_emit_music_status.await_args.kwargs["actor_name"] == "小芜"


@pytest.mark.asyncio
async def test_resume_music_while_agent_waiting_only_emits_user_join(monkeypatch):
    waiting = MusicCoListeningResponse(
        status="agent_waiting_user",
        track=MusicTrack(id="old-track", title="Old Track"),
        is_playing=False,
        initiated_by="user",
    )
    track = MusicTrack(id="track-1", title="Quiet Realm")
    monkeypatch.setattr(music, "get_open_co_listening", AsyncMock(return_value=waiting))
    monkeypatch.setattr(
        music,
        "upsert_now_playing",
        AsyncMock(return_value=(track, 12, True, "2026-06-11T12:00:00Z")),
    )
    monkeypatch.setattr(music_status, "persist_and_emit_music_status", AsyncMock(return_value="status-msg"))

    response = await music_api.update_music_now_playing(
        MusicPlaybackRequest(
            agent_id="agent-1",
            conversation_id="conv-1",
            track=MusicTrackPayload(id="track-1", title="Quiet Realm"),
            position_seconds=12,
            is_playing=True,
        ),
        user={"sub": "user-1"},
    )

    assert response.is_playing
    music_status.persist_and_emit_music_status.assert_awaited_once()
    assert music_status.persist_and_emit_music_status.await_args.kwargs["status"] == "started"
    assert music_status.persist_and_emit_music_status.await_args.kwargs["actor"] == "user"


@pytest.mark.asyncio
async def test_manual_track_change_in_active_co_listening_emits_reply(monkeypatch):
    active = MusicCoListeningResponse(
        status="active",
        track=MusicTrack(id="old-track", title="Old Track"),
        is_playing=True,
        initiated_by="user",
    )
    track = MusicTrack(id="track-1", title="Quiet Realm")
    monkeypatch.setattr(music, "get_open_co_listening", AsyncMock(return_value=active))
    monkeypatch.setattr(
        music,
        "upsert_now_playing",
        AsyncMock(return_value=(track, 12, True, "2026-06-11T12:00:00Z")),
    )
    monkeypatch.setattr(music_status, "persist_and_emit_music_status", AsyncMock())
    emit_reply = AsyncMock(return_value="assistant-msg")
    monkeypatch.setattr(music_status, "_emit_rendered_reply", emit_reply)
    monkeypatch.setattr(music_status, "_recent_music_prompt_exists", AsyncMock(return_value=False))

    response = await music_api.update_music_now_playing(
        MusicPlaybackRequest(
            agent_id="agent-1",
            conversation_id="conv-1",
            track=MusicTrackPayload(id="track-1", title="Quiet Realm"),
            position_seconds=12,
            is_playing=True,
            change_source="manual_next",
        ),
        user={"sub": "user-1"},
    )

    assert response.is_playing
    music_status.persist_and_emit_music_status.assert_not_awaited()
    emit_reply.assert_awaited_once()
    assert emit_reply.await_args.kwargs["prompt_key"] == "music.track_changed_manual"
    assert emit_reply.await_args.kwargs["music_co_listening"] is True


@pytest.mark.asyncio
async def test_auto_track_change_throttles_reply(monkeypatch):
    active = MusicCoListeningResponse(
        status="active",
        track=MusicTrack(id="old-track", title="Old Track"),
        is_playing=True,
        initiated_by="user",
    )
    track = MusicTrack(id="track-1", title="Quiet Realm")
    monkeypatch.setattr(music, "get_open_co_listening", AsyncMock(return_value=active))
    monkeypatch.setattr(
        music,
        "upsert_now_playing",
        AsyncMock(return_value=(track, 12, True, "2026-06-11T12:00:00Z")),
    )
    monkeypatch.setattr(music_status, "_recent_music_prompt_exists", AsyncMock(return_value=True))
    emit_reply = AsyncMock(return_value="assistant-msg")
    monkeypatch.setattr(music_status, "_emit_rendered_reply", emit_reply)

    await music_api.update_music_now_playing(
        MusicPlaybackRequest(
            agent_id="agent-1",
            conversation_id="conv-1",
            track=MusicTrackPayload(id="track-1", title="Quiet Realm"),
            position_seconds=12,
            is_playing=True,
            change_source="auto_next",
        ),
        user={"sub": "user-1"},
    )

    music_status._recent_music_prompt_exists.assert_awaited_once()
    assert music_status._recent_music_prompt_exists.await_args.kwargs["seconds"] == music_status.AUTO_TRACK_CHANGE_REPLY_SECONDS
    emit_reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_sync_track_change_does_not_emit_reply(monkeypatch):
    active = MusicCoListeningResponse(
        status="active",
        track=MusicTrack(id="old-track", title="Old Track"),
        is_playing=True,
        initiated_by="user",
    )
    track = MusicTrack(id="track-1", title="Quiet Realm")
    monkeypatch.setattr(music, "get_open_co_listening", AsyncMock(return_value=active))
    monkeypatch.setattr(
        music,
        "upsert_now_playing",
        AsyncMock(return_value=(track, 12, True, "2026-06-11T12:00:00Z")),
    )
    emit_reply = AsyncMock(return_value="assistant-msg")
    monkeypatch.setattr(music_status, "_emit_rendered_reply", emit_reply)

    await music_api.update_music_now_playing(
        MusicPlaybackRequest(
            agent_id="agent-1",
            conversation_id="conv-1",
            track=MusicTrackPayload(id="track-1", title="Quiet Realm"),
            position_seconds=12,
            is_playing=True,
            change_source="sync",
        ),
        user={"sub": "user-1"},
    )

    emit_reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_active_music_exits_when_agent_becomes_busy(monkeypatch):
    active = MusicCoListeningResponse(
        status="active",
        track=MusicTrack(id="track-1", title="Quiet Realm"),
        is_playing=True,
        initiated_by="user",
    )
    end_notice = AsyncMock(return_value=True)
    monkeypatch.setattr(music, "get_open_co_listening", AsyncMock(return_value=active))
    monkeypatch.setattr(music_status, "end_co_listening_with_notice", end_notice)
    monkeypatch.setattr(
        music_status,
        "_agent_join_status_exists",
        AsyncMock(return_value=True),
    )

    result = await music_status.reconcile_co_listening_for_status(
        user_id="user-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        workspace_id="ws-1",
        status_code="busy",
        activity="写报告",
        ai_name="小芜",
    )

    assert result is None
    end_notice.assert_awaited_once()
    assert end_notice.await_args.kwargs["prompt_key"] == "music.busy_exit"
    assert end_notice.await_args.kwargs["status_actor"] == "agent"
    assert end_notice.await_args.kwargs["status_actor_name"] == "小芜"


@pytest.mark.asyncio
async def test_reconcile_pending_music_does_not_emit_agent_busy_exit(monkeypatch):
    pending = MusicCoListeningResponse(
        status="pending_agent",
        track=MusicTrack(id="track-1", title="Quiet Realm"),
        is_playing=True,
        initiated_by="user",
    )
    end_notice = AsyncMock(return_value=True)
    monkeypatch.setattr(music, "get_open_co_listening", AsyncMock(return_value=pending))
    monkeypatch.setattr(music_status, "end_co_listening_with_notice", end_notice)

    result = await music_status.reconcile_co_listening_for_status(
        user_id="user-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        workspace_id="ws-1",
        status_code="busy",
        activity="写报告",
        ai_name="小芜",
    )

    assert result == pending
    end_notice.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_user_pending_active_music_does_not_emit_agent_busy_exit(monkeypatch):
    pending = MusicCoListeningResponse(
        status="active",
        track=MusicTrack(id="track-1", title="Quiet Realm"),
        is_playing=True,
        initiated_by="user_pending",
    )
    end_notice = AsyncMock(return_value=True)
    monkeypatch.setattr(music, "get_open_co_listening", AsyncMock(return_value=pending))
    monkeypatch.setattr(music_status, "end_co_listening_with_notice", end_notice)

    result = await music_status.reconcile_co_listening_for_status(
        user_id="user-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        workspace_id="ws-1",
        status_code="very_busy",
        activity="搬设备",
        ai_name="小青",
    )

    assert result == pending
    end_notice.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_legacy_active_user_without_agent_status_does_not_emit_busy_exit(monkeypatch):
    pending = MusicCoListeningResponse(
        status="active",
        track=MusicTrack(id="track-1", title="Quiet Realm"),
        is_playing=True,
        initiated_by="user",
    )
    end_notice = AsyncMock(return_value=True)
    monkeypatch.setattr(music, "get_open_co_listening", AsyncMock(return_value=pending))
    monkeypatch.setattr(music_status, "_agent_join_status_exists", AsyncMock(return_value=False))
    monkeypatch.setattr(music_status, "end_co_listening_with_notice", end_notice)

    result = await music_status.reconcile_co_listening_for_status(
        user_id="user-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        workspace_id="ws-1",
        status_code="busy",
        activity="搬设备",
        ai_name="小青",
    )

    assert result == pending
    end_notice.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_pending_stopped_music_emits_user_exit_and_late_missed(monkeypatch):
    pending = MusicCoListeningResponse(
        status="pending_agent",
        track=MusicTrack(id="track-1", title="Quiet Realm"),
        is_playing=False,
        initiated_by="user",
    )
    ended = pending.model_copy(update={"status": "ended"})
    end_session = AsyncMock(return_value=ended)
    emit_reply = AsyncMock(return_value="assistant-msg")
    persist_status = AsyncMock(return_value="status-msg")
    monkeypatch.setattr(music, "get_open_co_listening", AsyncMock(return_value=pending))
    monkeypatch.setattr(music, "end_co_listening", end_session)
    monkeypatch.setattr(music_status, "_emit_rendered_reply", emit_reply)
    monkeypatch.setattr(music_status, "persist_and_emit_music_status", persist_status)

    result = await music_status.reconcile_co_listening_for_status(
        user_id="user-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        workspace_id="ws-1",
        status_code="idle",
        activity="自由时间",
        ai_name="小芜",
    )

    assert result is None
    end_session.assert_awaited_once()
    assert end_session.await_args.kwargs["reason"] == "user_stopped_before_agent_join"
    persist_status.assert_awaited_once()
    assert persist_status.await_args.kwargs["status"] == "ended"
    assert persist_status.await_args.kwargs["actor"] == "user"
    assert persist_status.await_args.kwargs["reason"] == "user_stopped_before_agent_join"
    emit_reply.assert_awaited_once()
    assert emit_reply.await_args.kwargs["prompt_key"] == "music.agent_late_missed"


@pytest.mark.asyncio
async def test_reconcile_late_missed_music_has_no_agent_exit_status(monkeypatch):
    missed = MusicCoListeningResponse(
        status="ended",
        track=MusicTrack(id="track-1", title="Quiet Realm"),
        is_playing=False,
        initiated_by="user",
        ended_reason="user_pause_timeout_before_agent_join",
    )
    emit_reply = AsyncMock(return_value="assistant-msg")
    persist_status = AsyncMock(return_value="status-msg")
    ack = AsyncMock()
    monkeypatch.setattr(music, "get_open_co_listening", AsyncMock(return_value=None))
    monkeypatch.setattr(
        music,
        "get_recent_unacknowledged_user_music_stop",
        AsyncMock(return_value=missed),
    )
    monkeypatch.setattr(music, "mark_user_music_stop_acknowledged", ack)
    monkeypatch.setattr(music_status, "_emit_rendered_reply", emit_reply)
    monkeypatch.setattr(music_status, "persist_and_emit_music_status", persist_status)

    result = await music_status.reconcile_co_listening_for_status(
        user_id="user-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        workspace_id="ws-1",
        status_code="idle",
        activity="自由时间",
        ai_name="小芜",
    )

    assert result is None
    emit_reply.assert_awaited_once()
    assert emit_reply.await_args.kwargs["prompt_key"] == "music.agent_late_missed"
    ack.assert_awaited_once()
    persist_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_recent_user_music_stop_only_matches_before_agent_join_reasons(monkeypatch):
    fake_db = _FakeDb()
    monkeypatch.setattr(music, "db", fake_db)

    result = await music.get_recent_unacknowledged_user_music_stop(
        conversation_id="conv-1",
    )

    assert result is None
    query = fake_db.queries[0][0]
    assert "user_pause_timeout_before_agent_join" in query
    assert "user_exit_before_agent_join" in query
    assert "connection_lost_before_agent_join" in query
    assert "'user_pause_timeout'," not in query
    assert "'user_exit'," not in query
    assert "'connection_lost'" not in query


@pytest.mark.asyncio
async def test_pause_timeout_pending_music_emits_status_without_agent_reply(monkeypatch):
    pending = MusicCoListeningResponse(
        status="pending_agent",
        track=MusicTrack(id="track-1", title="Quiet Realm"),
        is_playing=False,
        initiated_by="user",
    )
    ended = pending.model_copy(update={
        "status": "ended",
        "ended_reason": "user_pause_timeout_before_agent_join",
    })
    render_reply = AsyncMock(return_value="不该发送")
    persist_message = AsyncMock(return_value="assistant-msg")
    persist_status = AsyncMock(return_value="status-msg")
    monkeypatch.setattr(music, "get_open_co_listening", AsyncMock(return_value=pending))
    monkeypatch.setattr(music, "end_paused_co_listening_if_stale", AsyncMock(return_value=ended))
    monkeypatch.setattr(music_status, "_render_exit_reply", render_reply)
    monkeypatch.setattr(music_status, "_persist_assistant_message", persist_message)
    monkeypatch.setattr(music_status, "persist_and_emit_music_status", persist_status)

    await music_status.end_if_paused_after_timeout(
        user_id="user-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        seconds=0,
    )

    render_reply.assert_not_awaited()
    persist_message.assert_not_awaited()
    persist_status.assert_awaited_once()
    assert persist_status.await_args.kwargs["actor"] == "user"
    assert persist_status.await_args.kwargs["reason"] == "user_pause_timeout_before_agent_join"


@pytest.mark.asyncio
async def test_end_music_co_listening_pending_pause_has_no_agent_reply(monkeypatch):
    pending = MusicCoListeningResponse(
        status="pending_agent",
        track=MusicTrack(id="track-1", title="Quiet Realm"),
        is_playing=False,
        initiated_by="user",
    )
    end_with_notice = AsyncMock(return_value=True)
    end_session = AsyncMock(return_value=pending.model_copy(update={"status": "ended"}))
    persist_status = AsyncMock(return_value="status-msg")
    monkeypatch.setattr(music, "get_open_co_listening", AsyncMock(return_value=pending))
    monkeypatch.setattr(music, "end_co_listening", end_session)
    monkeypatch.setattr(music_status, "end_co_listening_with_notice", end_with_notice)
    monkeypatch.setattr(music_status, "persist_and_emit_music_status", persist_status)

    await music_api.end_music_co_listening(
        MusicCoListeningEndRequest(
            agent_id="agent-1",
            conversation_id="conv-1",
            reason="user_pause_timeout",
        ),
        user={"sub": "user-1"},
    )

    end_with_notice.assert_not_awaited()
    end_session.assert_awaited_once()
    assert end_session.await_args.kwargs["reason"] == "user_pause_timeout_before_agent_join"
    persist_status.assert_awaited_once()
    assert persist_status.await_args.kwargs["actor"] == "user"
    assert persist_status.await_args.kwargs["reason"] == "user_pause_timeout_before_agent_join"


@pytest.mark.asyncio
async def test_end_music_co_listening_pause_is_idempotent_while_agent_waiting(monkeypatch):
    waiting = MusicCoListeningResponse(
        status="agent_waiting_user",
        track=MusicTrack(id="track-1", title="Quiet Realm"),
        is_playing=False,
        initiated_by="user",
    )
    end_session = AsyncMock(return_value=waiting.model_copy(update={"status": "ended"}))
    persist_status = AsyncMock(return_value="status-msg")
    monkeypatch.setattr(music, "get_open_co_listening", AsyncMock(return_value=waiting))
    monkeypatch.setattr(music, "end_co_listening", end_session)
    monkeypatch.setattr(music_status, "persist_and_emit_music_status", persist_status)

    await music_api.end_music_co_listening(
        MusicCoListeningEndRequest(
            agent_id="agent-1",
            conversation_id="conv-1",
            reason="user_pause_timeout",
        ),
        user={"sub": "user-1"},
    )

    end_session.assert_not_awaited()
    persist_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_disconnect_pending_music_emits_user_exit_status(monkeypatch):
    pending = MusicCoListeningResponse(
        status="pending_agent",
        track=MusicTrack(id="track-1", title="Quiet Realm"),
        is_playing=True,
        initiated_by="user",
    )
    ended = pending.model_copy(update={
        "status": "ended",
        "ended_reason": "connection_lost_before_agent_join",
    })
    persist_status = AsyncMock(return_value="status-msg")
    monkeypatch.setattr(music_status.manager, "get", lambda _conversation_id: None)
    monkeypatch.setattr(music, "get_open_co_listening", AsyncMock(return_value=pending))
    monkeypatch.setattr(music, "end_co_listening", AsyncMock(return_value=ended))
    monkeypatch.setattr(music_status, "persist_and_emit_music_status", persist_status)

    await music_status.end_if_disconnected_after_timeout(
        user_id="user-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        seconds=0,
    )

    music.end_co_listening.assert_awaited_once()
    assert music.end_co_listening.await_args.kwargs["reason"] == "connection_lost_before_agent_join"
    persist_status.assert_awaited_once()
    assert persist_status.await_args.kwargs["status"] == "ended"
    assert persist_status.await_args.kwargs["actor"] == "user"
    assert persist_status.await_args.kwargs["reason"] == "connection_lost_before_agent_join"
