from __future__ import annotations

import pytest
from starlette.requests import Request

from app.api.public import music as music_api
from app.models.music import MusicTrackPayload
from app.services import music


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
