"""Unit tests for app.api.ownership dependency helpers.

Verifies:
- require_user_self: path/query user_id != JWT sub → 403
- require_conversation_owner: missing → 404, wrong owner → 403, hit → returns
- require_memory_owner: missing → 404, wrong owner → 403, hit → returns
- require_agent_owner (existing): wrong owner → 403, archived → 404
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from app.api.ownership import (
    require_agent_owner,
    require_conversation_owner,
    require_memory_owner,
    require_user_self,
)


# ─────────────────── require_user_self ───────────────────

@pytest.mark.asyncio
async def test_require_user_self_passes_when_match():
    user = {"sub": "u1", "role": "user"}
    result = await require_user_self(user_id="u1", user=user)
    assert result is user


@pytest.mark.asyncio
async def test_require_user_self_rejects_when_mismatch():
    user = {"sub": "u1", "role": "user"}
    with pytest.raises(HTTPException) as exc:
        await require_user_self(user_id="other-user", user=user)
    assert exc.value.status_code == 403
    assert "Not your data" in exc.value.detail


@pytest.mark.asyncio
async def test_require_user_self_rejects_admin_role_too():
    """admin role 也不能在 public endpoint 读他人数据 (走 admin-api)."""
    user = {"sub": "u1", "role": "admin"}
    with pytest.raises(HTTPException) as exc:
        await require_user_self(user_id="other-user", user=user)
    assert exc.value.status_code == 403


# ─────────────────── require_conversation_owner ───────────────────

@pytest.mark.asyncio
async def test_require_conversation_owner_404_when_missing():
    fake_db = AsyncMock()
    fake_db.conversation.find_unique = AsyncMock(return_value=None)
    user = {"sub": "u1", "role": "user"}
    with patch("app.api.ownership.db", fake_db):
        with pytest.raises(HTTPException) as exc:
            await require_conversation_owner(conversation_id="conv-1", user=user)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_require_conversation_owner_404_when_deleted():
    conv = SimpleNamespace(id="conv-1", userId="u1", isDeleted=True)
    fake_db = AsyncMock()
    fake_db.conversation.find_unique = AsyncMock(return_value=conv)
    user = {"sub": "u1", "role": "user"}
    with patch("app.api.ownership.db", fake_db):
        with pytest.raises(HTTPException) as exc:
            await require_conversation_owner(conversation_id="conv-1", user=user)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_require_conversation_owner_403_when_wrong_owner():
    conv = SimpleNamespace(id="conv-1", userId="other-user", isDeleted=False)
    fake_db = AsyncMock()
    fake_db.conversation.find_unique = AsyncMock(return_value=conv)
    user = {"sub": "u1", "role": "user"}
    with patch("app.api.ownership.db", fake_db):
        with pytest.raises(HTTPException) as exc:
            await require_conversation_owner(conversation_id="conv-1", user=user)
    assert exc.value.status_code == 403
    assert "Not your conversation" in exc.value.detail


@pytest.mark.asyncio
async def test_require_conversation_owner_returns_conv_when_owner():
    conv = SimpleNamespace(id="conv-1", userId="u1", isDeleted=False)
    fake_db = AsyncMock()
    fake_db.conversation.find_unique = AsyncMock(return_value=conv)
    user = {"sub": "u1", "role": "user"}
    with patch("app.api.ownership.db", fake_db):
        result = await require_conversation_owner(conversation_id="conv-1", user=user)
    assert result is conv


# ─────────────────── require_memory_owner ───────────────────

@pytest.mark.asyncio
async def test_require_memory_owner_404_when_missing():
    user = {"sub": "u1", "role": "user"}
    with patch(
        "app.services.memory.storage.repo.find_unique",
        new_callable=AsyncMock, return_value=None,
    ):
        with pytest.raises(HTTPException) as exc:
            await require_memory_owner(memory_id="mem-1", user=user)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_require_memory_owner_403_when_wrong_owner():
    m = SimpleNamespace(id="mem-1", userId="other-user")
    user = {"sub": "u1", "role": "user"}
    with patch(
        "app.services.memory.storage.repo.find_unique",
        new_callable=AsyncMock, return_value=m,
    ):
        with pytest.raises(HTTPException) as exc:
            await require_memory_owner(memory_id="mem-1", user=user)
    assert exc.value.status_code == 403
    assert "Not your memory" in exc.value.detail


@pytest.mark.asyncio
async def test_require_memory_owner_returns_memory_when_owner():
    m = SimpleNamespace(id="mem-1", userId="u1")
    user = {"sub": "u1", "role": "user"}
    with patch(
        "app.services.memory.storage.repo.find_unique",
        new_callable=AsyncMock, return_value=m,
    ):
        result = await require_memory_owner(memory_id="mem-1", user=user)
    assert result is m


# ─────────────────── require_agent_owner_any_status ───────────────────

@pytest.mark.asyncio
async def test_require_agent_owner_any_status_404_when_missing():
    from app.api.ownership import require_agent_owner_any_status
    fake_db = AsyncMock()
    fake_db.aiagent.find_unique = AsyncMock(return_value=None)
    user = {"sub": "u1", "role": "user"}
    with patch("app.api.ownership.db", fake_db):
        with pytest.raises(HTTPException) as exc:
            await require_agent_owner_any_status(agent_id="a-1", user=user)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_require_agent_owner_any_status_passes_when_provisioning():
    """provisioning 状态也能通过 (用于 provision-status endpoint)."""
    from app.api.ownership import require_agent_owner_any_status
    agent = SimpleNamespace(id="a-1", userId="u1", status="provisioning")
    fake_db = AsyncMock()
    fake_db.aiagent.find_unique = AsyncMock(return_value=agent)
    user = {"sub": "u1", "role": "user"}
    with patch("app.api.ownership.db", fake_db):
        result = await require_agent_owner_any_status(agent_id="a-1", user=user)
    assert result is agent


@pytest.mark.asyncio
async def test_require_agent_owner_any_status_403_when_wrong_owner():
    from app.api.ownership import require_agent_owner_any_status
    agent = SimpleNamespace(id="a-1", userId="other-user", status="provisioning")
    fake_db = AsyncMock()
    fake_db.aiagent.find_unique = AsyncMock(return_value=agent)
    user = {"sub": "u1", "role": "user"}
    with patch("app.api.ownership.db", fake_db):
        with pytest.raises(HTTPException) as exc:
            await require_agent_owner_any_status(agent_id="a-1", user=user)
    assert exc.value.status_code == 403


# ─────────────────── require_agent_owner (active-only) ───────────────────

@pytest.mark.asyncio
async def test_require_agent_owner_404_when_archived():
    """archived agent 即使是 owner 也走不通 active-only dep."""
    agent = SimpleNamespace(id="a-1", userId="u1", status="archived")
    with pytest.raises(HTTPException) as exc:
        await require_agent_owner(agent=agent)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_require_agent_owner_passes_when_active():
    agent = SimpleNamespace(id="a-1", userId="u1", status="active")
    result = await require_agent_owner(agent=agent)
    assert result is agent
