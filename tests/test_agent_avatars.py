import hashlib
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api.public import agents as agents_api
from app.services import agent_avatars
from app.services.agent_template import clone as agent_clone


def test_agent_response_uses_bundled_avatar_url():
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
        avatarKey="companion-female-01",
        avatarUrl="https://api.dicebear.com/old.png",
        createdAt="2026-01-01T00:00:00",
    )

    response = agents_api._agent_response(agent)

    assert response.avatar_key == "companion-female-01"
    assert response.avatar_url == "/agents/avatar/companion-female-01.png"


def test_avatar_keys_for_gender_cover_bundled_pool():
    male_keys = agent_avatars.avatar_keys_for_gender("male")
    female_keys = agent_avatars.avatar_keys_for_gender("female")

    assert len(male_keys) == 24
    assert len(female_keys) == 21
    assert male_keys[-1] == "companion-male-27"
    assert "companion-male-14" not in male_keys
    assert "companion-male-17" not in male_keys
    assert "companion-male-24" not in male_keys
    assert "companion-female-21" not in female_keys
    assert all("-male-" in key for key in male_keys)
    assert all("-female-" in key for key in female_keys)
    assert len(agent_avatars.avatar_keys_for_gender(None)) == 45


def test_every_avatar_key_resolves_to_a_bundled_png():
    agent_avatars.validate_avatar_assets()

    keys = agent_avatars.avatar_keys_for_gender()
    paths = [agent_avatars.get_avatar_path(key) for key in keys]
    for key, path in zip(keys, paths, strict=True):
        assert path.name == f"{key}.png"
        assert path.stat().st_size > 0

    assert {path.stem for path in paths[0].parent.glob("*.png")} == set(keys)
    digests = {hashlib.sha256(path.read_bytes()).digest() for path in paths}
    assert len(digests) == len(paths)


@pytest.mark.asyncio
async def test_avatar_endpoint_serves_bundled_png_with_immutable_cache():
    response = await agents_api.get_agent_avatar("companion-male-01")

    assert response.path == agent_avatars.get_avatar_path("companion-male-01")
    assert response.media_type == "image/png"
    assert response.headers["cache-control"] == "public, max-age=31536000, immutable"


@pytest.mark.parametrize(
    "key",
    [
        "../secret",
        "companion-male-14",
        "companion-female-21",
        "companion-male-99",
        "bansheng-male-01",
    ],
)
def test_avatar_path_rejects_unknown_keys(key):
    with pytest.raises(HTTPException) as exc_info:
        agent_avatars.get_avatar_path(key)

    assert exc_info.value.status_code == 404


def test_template_clone_selects_a_fresh_gender_matched_avatar(monkeypatch):
    observed: list[str | None] = []

    def fake_pick(gender: str | None):
        observed.append(gender)
        return agent_avatars.AgentAvatar(
            key="companion-female-07",
            url="/agents/avatar/companion-female-07.png",
        )

    monkeypatch.setattr(agent_clone, "pick_agent_avatar", fake_pick)
    template = SimpleNamespace(
        name="小伴",
        background=None,
        lifeOverview=None,
        age=None,
        occupation=None,
        city=None,
        gender="female",
        mbti=None,
        currentMbti=None,
        values=None,
    )

    payload = agent_clone._clone_persona_data(template, "user-id")

    assert observed == ["female"]
    assert payload["avatarKey"] == "companion-female-07"
    assert payload["avatarUrl"] == "/agents/avatar/companion-female-07.png"
