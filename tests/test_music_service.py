from __future__ import annotations

import pytest

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
        library="audio.focus",
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
        library="audio.focus",
        limit=1,
        refresh=True,
    )

    assert use_cache_values == [False]


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
            artist="AudioLib",
            album="AudioLib Focus",
            library="audio.focus",
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
                "data": {
                    "title": "Deep Space Drift",
                    "url": "https://cdn.example.test/deep-space.mp3",
                    "duration_sec": 240,
                }
            }

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, *args, **kwargs):
            calls.append((args, kwargs))
            return _FakeResponse()

    music._audio_cache.clear()
    monkeypatch.setattr(music.settings, "audiolib_api_key", "alp_test_key")
    monkeypatch.setattr(music.settings, "audiolib_base_url", "https://api.audiolib.ai")
    monkeypatch.setattr(music.httpx, "AsyncClient", _FakeAsyncClient)

    first = await music.fetch_random_track("audio.focus", index=0)
    second = await music.fetch_random_track("audio.focus", index=0)

    assert first.id == second.id
    assert first.url == "https://cdn.example.test/deep-space.mp3"
    assert len(calls) == 1


def test_mock_track_cover_uses_track_identity_not_list_index():
    focus = music._mock_track("audio.focus", index=0)
    relax = music._mock_track("audio.relax", index=0)

    assert focus.cover_key != relax.cover_key
