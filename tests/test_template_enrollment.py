"""Open/stop templates independently; new users pick randomly from the open pool.

Stopping a template must never archive sibling templates or cloned user agents.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.agent_template import clone as clone_mod
from app.services.agent_template import registry


def test_is_enrolling_requires_active_and_flag():
    assert registry.is_enrolling(SimpleNamespace(status="active", templateEnabled=True))
    assert not registry.is_enrolling(SimpleNamespace(status="active", templateEnabled=False))
    assert not registry.is_enrolling(SimpleNamespace(status="archived", templateEnabled=True))
    assert not registry.is_enrolling(SimpleNamespace(status="provisioning", templateEnabled=True))
    # Old Prisma client without the field → provisioned templates stay open.
    assert registry.is_enrolling(SimpleNamespace(status="active"))


def test_pick_enrolling_template_id():
    assert registry.pick_enrolling_template_id([]) is None
    assert registry.pick_enrolling_template_id(["only"]) == "only"
    with patch.object(registry.secrets, "choice", return_value="b") as choice:
        assert registry.pick_enrolling_template_id(["a", "b", "c"]) == "b"
        choice.assert_called_once_with(["a", "b", "c"])


@pytest.mark.asyncio
async def test_disable_only_flips_the_flag():
    execute = AsyncMock()
    restore = AsyncMock()
    oversized = AsyncMock()
    with patch.object(registry.db, "execute_raw", execute), \
         patch.object(registry, "_restore_template_runtime", restore), \
         patch.object(registry, "count_oversized_memories", oversized):
        await registry.set_template_enabled("tpl-1", False)

    restore.assert_not_awaited()
    oversized.assert_not_awaited()
    execute.assert_awaited_once()
    sql, enabled, agent_id = execute.await_args.args
    assert "template_enabled" in sql
    assert enabled is False
    assert agent_id == "tpl-1"


@pytest.mark.asyncio
async def test_enable_refuses_oversized_persona_and_does_not_open():
    execute = AsyncMock()
    restore = AsyncMock()
    with patch.object(registry.db, "execute_raw", execute), \
         patch.object(registry, "_restore_template_runtime", restore), \
         patch.object(registry, "count_oversized_memories", AsyncMock(return_value=2)):
        with pytest.raises(ValueError, match="不能开放给新用户"):
            await registry.set_template_enabled("tpl-dirty", True)

    restore.assert_awaited_once_with("tpl-dirty")
    execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_enable_restores_only_this_template_runtime():
    """A wrongly-archived template can be reopened; clones are not queried."""
    execute = AsyncMock()
    agent_update = AsyncMock()
    reactivate = AsyncMock()
    fake_aiagent = MagicMock()
    fake_aiagent.find_unique = AsyncMock(
        return_value=SimpleNamespace(status="archived", id="tpl-1"),
    )
    fake_aiagent.update = agent_update
    fake_ws = MagicMock()
    fake_ws.find_first = AsyncMock(
        return_value=SimpleNamespace(id="ws-tpl", status="archived"),
    )

    with patch.object(registry, "count_oversized_memories", AsyncMock(return_value=0)), \
         patch.object(registry.db, "execute_raw", execute), \
         patch.object(registry.db, "aiagent", fake_aiagent), \
         patch.object(registry.db, "chatworkspace", fake_ws), \
         patch(
             "app.services.workspace.workspaces.reactivate_workspace",
             reactivate,
         ):
        await registry.set_template_enabled("tpl-1", True)

    agent_update.assert_awaited_once()
    assert agent_update.await_args.kwargs["where"] == {"id": "tpl-1"}
    assert agent_update.await_args.kwargs["data"]["status"] == "active"
    reactivate.assert_awaited_once_with("ws-tpl")
    execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_clones_the_picked_open_template(monkeypatch):
    monkeypatch.setattr(
        clone_mod, "list_enrolling_template_ids",
        AsyncMock(return_value=["open-1", "open-2"]),
    )
    monkeypatch.setattr(clone_mod, "pick_enrolling_template_id", lambda ids: "open-2")
    monkeypatch.setattr(clone_mod, "_has_agent_or_pending", AsyncMock(return_value=False))
    cloned = object()
    clone_fn = AsyncMock(return_value=(cloned, object(), object()))
    monkeypatch.setattr(clone_mod, "clone_template_agent_for_user", clone_fn)

    async def _no_redis():
        raise RuntimeError("redis down")

    monkeypatch.setattr("app.redis_client.get_redis", _no_redis)

    result = await clone_mod.ensure_default_agent_for_user("user-1")
    assert result is cloned
    clone_fn.assert_awaited_once_with("user-1", "open-2")


@pytest.mark.asyncio
async def test_ensure_noop_when_pool_empty(monkeypatch):
    monkeypatch.setattr(
        clone_mod, "list_enrolling_template_ids", AsyncMock(return_value=[]),
    )
    clone_fn = AsyncMock()
    monkeypatch.setattr(clone_mod, "clone_template_agent_for_user", clone_fn)
    assert await clone_mod.ensure_default_agent_for_user("user-1") is None
    clone_fn.assert_not_awaited()
