"""H5 onboarding: claim a gender-matched open template after login."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.api.public import auth as auth_api
from app.models.auth import AuthResponse, ClaimH5AgentRequest


def _user():
    return SimpleNamespace(id="user-1", username="wx_user", role="user")


def _patch_user(monkeypatch, user):
    monkeypatch.setattr(
        auth_api.db,
        "user",
        SimpleNamespace(find_unique=AsyncMock(return_value=user)),
    )


def _expected(*, has_agent: bool = True) -> AuthResponse:
    return AuthResponse(
        token="jwt",
        user_id="user-1",
        username="wx_user",
        role="user",
        has_agent=has_agent,
    )


@pytest.mark.asyncio
async def test_claim_h5_agent_clones_matching_gender(monkeypatch):
    _patch_user(monkeypatch, _user())
    monkeypatch.setattr(auth_api, "get_active_workspace", AsyncMock(return_value=None))
    monkeypatch.setattr(
        auth_api, "list_enrolling_template_ids", AsyncMock(return_value=["tpl-f"]),
    )
    clone = AsyncMock(return_value=object())
    monkeypatch.setattr(auth_api, "ensure_default_agent_for_user", clone)
    monkeypatch.setattr(auth_api, "create_jwt", lambda user_id, role: "jwt")
    monkeypatch.setattr(
        auth_api, "_build_auth_response", AsyncMock(return_value=_expected()),
    )

    response = await auth_api.claim_h5_agent(
        ClaimH5AgentRequest(gender="female"),
        {"sub": "user-1"},
    )

    assert response.has_agent is True
    auth_api.list_enrolling_template_ids.assert_awaited_once_with(gender="female")
    clone.assert_awaited_once_with("user-1", "female")


@pytest.mark.asyncio
async def test_claim_h5_agent_is_noop_when_user_already_has_agent(monkeypatch):
    _patch_user(monkeypatch, _user())
    monkeypatch.setattr(
        auth_api, "get_active_workspace", AsyncMock(return_value=object()),
    )
    listed = AsyncMock()
    clone = AsyncMock()
    monkeypatch.setattr(auth_api, "list_enrolling_template_ids", listed)
    monkeypatch.setattr(auth_api, "ensure_default_agent_for_user", clone)
    monkeypatch.setattr(auth_api, "create_jwt", lambda user_id, role: "jwt")
    monkeypatch.setattr(
        auth_api, "_build_auth_response", AsyncMock(return_value=_expected()),
    )

    await auth_api.claim_h5_agent(
        ClaimH5AgentRequest(gender="male"),
        {"sub": "user-1"},
    )

    listed.assert_not_awaited()
    clone.assert_not_awaited()


@pytest.mark.asyncio
async def test_claim_h5_agent_rejects_empty_gender_pool(monkeypatch):
    _patch_user(monkeypatch, _user())
    monkeypatch.setattr(auth_api, "get_active_workspace", AsyncMock(return_value=None))
    monkeypatch.setattr(
        auth_api, "list_enrolling_template_ids", AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(auth_api, "create_jwt", lambda user_id, role: "jwt")

    with pytest.raises(HTTPException) as exc:
        await auth_api.claim_h5_agent(
            ClaimH5AgentRequest(gender="male"),
            {"sub": "user-1"},
        )
    assert exc.value.status_code == 400
    assert "性别" in exc.value.detail


@pytest.mark.asyncio
async def test_claim_h5_agent_errors_when_clone_fails(monkeypatch):
    _patch_user(monkeypatch, _user())
    monkeypatch.setattr(auth_api, "get_active_workspace", AsyncMock(return_value=None))
    monkeypatch.setattr(
        auth_api, "list_enrolling_template_ids", AsyncMock(return_value=["tpl-m"]),
    )
    monkeypatch.setattr(
        auth_api, "ensure_default_agent_for_user", AsyncMock(return_value=None),
    )
    monkeypatch.setattr(auth_api, "create_jwt", lambda user_id, role: "jwt")

    with pytest.raises(HTTPException) as exc:
        await auth_api.claim_h5_agent(
            ClaimH5AgentRequest(gender="male"),
            {"sub": "user-1"},
        )
    assert exc.value.status_code == 500
    assert "创建" in exc.value.detail
