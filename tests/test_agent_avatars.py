from types import SimpleNamespace
import base64

import pytest
from fastapi import HTTPException

from app.api.public import agents as agents_api
from app.services import agent_avatars


class _FakeResponse:
    def __init__(self, content: bytes = b"png-bytes", content_type: str = "image/png"):
        self.content = content
        self.headers = {"content-type": content_type}

    def raise_for_status(self) -> None:
        return None


class _FakeAsyncClient:
    calls: list[str] = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def get(self, url: str):
        self.calls.append(url)
        return _FakeResponse()


class _FakeAvatarCacheDelegate:
    def __init__(self):
        self.rows: dict[str, SimpleNamespace] = {}

    async def find_unique(self, *, where):
        return self.rows.get(where["key"])

    async def upsert(self, *, where, data):
        key = where["key"]
        payload = data["update"] if key in self.rows else data["create"]
        row = SimpleNamespace(
            key=key,
            gender=payload["gender"],
            contentType=payload["contentType"],
            imageBytes=payload["imageBytes"],
            sourceUrl=payload["sourceUrl"],
        )
        self.rows[key] = row
        return row


def test_agent_response_uses_cached_avatar_url():
    agent = SimpleNamespace(
        id="agent-id",
        name="TestBot",
        userId="user-id",
        mbti=None,
        currentMbti=None,
        background=None,
        values=None,
        gender="female",
        city=None,
        lifeOverview=None,
        avatarKey="bansheng-female-01",
        avatarUrl="https://api.dicebear.com/old.png",
        createdAt="2026-01-01T00:00:00",
    )

    response = agents_api._agent_response(agent)

    assert response.avatar_key == "bansheng-female-01"
    assert response.avatar_url == "/agents/avatar/bansheng-female-01.png"


@pytest.mark.asyncio
async def test_ensure_cached_avatar_downloads_and_reuses_db_cache(monkeypatch):
    cache = _FakeAvatarCacheDelegate()
    monkeypatch.setattr(agent_avatars.db, "agentavatarcache", cache, raising=False)
    monkeypatch.setattr(agent_avatars.httpx, "AsyncClient", _FakeAsyncClient)
    _FakeAsyncClient.calls = []

    first = await agent_avatars.ensure_cached_avatar("bansheng-female-01")
    second = await agent_avatars.ensure_cached_avatar("bansheng-female-01")

    assert first == second
    assert first.image_bytes == b"png-bytes"
    assert first.content_type == "image/png"
    assert cache.rows["bansheng-female-01"].gender == "female"
    assert cache.rows["bansheng-female-01"].imageBytes == base64.b64encode(b"png-bytes").decode(
        "ascii"
    )
    assert len(_FakeAsyncClient.calls) == 1


def test_avatar_keys_for_gender():
    assert all("-male-" in key for key in agent_avatars.avatar_keys_for_gender("male"))
    assert all("-female-" in key for key in agent_avatars.avatar_keys_for_gender("female"))
    assert len(agent_avatars.avatar_keys_for_gender(None)) == 12


def test_avatar_from_row_accepts_base64_image_bytes():
    row = SimpleNamespace(
        key="bansheng-female-01",
        contentType="image/png",
        imageBytes=base64.b64encode(b"png-bytes").decode("ascii"),
    )

    avatar = agent_avatars._avatar_from_row(row)

    assert avatar is not None
    assert avatar.image_bytes == b"png-bytes"


def test_build_cached_avatar_url_rejects_path_like_keys():
    with pytest.raises(HTTPException):
        agent_avatars.build_cached_avatar_url("../secret")
