"""From-scratch agent creation samples a gender-matched library name.

Flutter no longer sends a placeholder. The sampled name is the value stored
on the agent, which the init job feeds into persona generation and the
「我叫…」 L1 memory.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from app.api.public.agents import _resolve_create_name
from tests.conftest import make_auth_header

_PERSONALITY = {
    "lively": 70,
    "rational": 60,
    "emotional": 40,
    "planned": 55,
    "spontaneous": 45,
    "creative": 65,
    "humor": 50,
}


def _agent(name: str, gender: str = "male"):
    return SimpleNamespace(
        id="agent-id",
        name=name,
        userId="user-id",
        mbti=None,
        currentMbti=None,
        background=None,
        values=None,
        gender=gender,
        lifeOverview=None,
        createdAt="2025-01-01T00:00:00",
    )


@pytest.mark.asyncio
async def test_blank_name_samples_library_by_gender():
    with patch(
        "app.api.public.agents.pick_random_names",
        new_callable=AsyncMock,
        return_value=[{"name": "乔慕川", "gender": "male"}],
    ) as pick:
        assert await _resolve_create_name("  ", "male") == "乔慕川"
    pick.assert_awaited_once_with("male", 1)


@pytest.mark.asyncio
async def test_explicit_name_is_not_replaced():
    with patch(
        "app.api.public.agents.pick_random_names",
        new_callable=AsyncMock,
    ) as pick:
        assert await _resolve_create_name(" 林昭 ", "female") == "林昭"
    pick.assert_not_awaited()


@pytest.mark.asyncio
async def test_blank_name_without_gender_is_rejected():
    with patch(
        "app.api.public.agents.pick_random_names",
        new_callable=AsyncMock,
    ) as pick:
        with pytest.raises(HTTPException) as exc:
            await _resolve_create_name(None, None)
    assert exc.value.status_code == 400
    pick.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_name_pool_is_a_client_error():
    with patch(
        "app.api.public.agents.pick_random_names",
        new_callable=AsyncMock,
        side_effect=ValueError("该性别姓名库可用名字不足 1 个"),
    ):
        with pytest.raises(HTTPException) as exc:
            await _resolve_create_name(None, "女")
    assert exc.value.status_code == 400
    assert "姓名库" in exc.value.detail


def test_create_agent_omitted_name_uses_sampled_name(api_client):
    client = api_client
    agent = _agent("乔慕川")
    workspace = SimpleNamespace(id="workspace-id")
    with (
        patch(
            "app.api.public.agents.pick_random_names",
            new_callable=AsyncMock,
            return_value=[{"name": "乔慕川", "gender": "male"}],
        ) as pick,
        patch(
            "app.api.public.agents.create_agent_with_provisioning",
            new_callable=AsyncMock,
            return_value=(agent, workspace),
        ) as create,
    ):
        response = client.post(
            "/agents",
            headers=make_auth_header("user-id"),
            json={
                "user_id": "user-id",
                "gender": "male",
                "personality": _PERSONALITY,
            },
        )
    assert response.status_code == 200
    assert response.json()["name"] == "乔慕川"
    pick.assert_awaited_once_with("male", 1)
    assert create.await_args.kwargs["name"] == "乔慕川"
    assert create.await_args.kwargs["gender"] == "male"


def test_create_agent_normalizes_chinese_gender_before_sampling(api_client):
    client = api_client
    agent = _agent("沈知远", gender="male")
    workspace = SimpleNamespace(id="workspace-id")
    with (
        patch(
            "app.api.public.agents.pick_random_names",
            new_callable=AsyncMock,
            return_value=[{"name": "沈知远", "gender": "male"}],
        ) as pick,
        patch(
            "app.api.public.agents.create_agent_with_provisioning",
            new_callable=AsyncMock,
            return_value=(agent, workspace),
        ) as create,
    ):
        response = client.post(
            "/agents",
            headers=make_auth_header("user-id"),
            json={
                "user_id": "user-id",
                "gender": "男",
                "personality": _PERSONALITY,
            },
        )
    assert response.status_code == 200
    pick.assert_awaited_once_with("male", 1)
    assert create.await_args.kwargs["gender"] == "male"
    assert create.await_args.kwargs["name"] == "沈知远"
