"""Creating a new template must not archive the template user's other templates.

The template system user owns MANY coexisting templates (one is the default).
create_agent_with_provisioning was built for the normal single-companion model,
so it staged (archived) the user's other active workspaces — which archived
every previously-created template, hiding them from the admin list and making
them undeletable. The template path now passes stage_existing_workspaces=False.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.public import agents as agents_mod
from app.services.agent_template import registry as reg


class _FakeAgent:
    def __init__(self, id="tpl-new"):
        self.id = id
        self.userId = "sys-owner"
        self.gender = "female"
        self.avatarKey = None


def _fake_ws():
    ws = MagicMock()
    ws.id = "ws-new"
    return ws


def _provisioning_patches(fake_aiagent, stage, finalize):
    return (
        patch.object(agents_mod.db, "aiagent", fake_aiagent),
        patch.object(agents_mod, "create_provisioning_workspace", AsyncMock(return_value=_fake_ws())),
        patch.object(agents_mod, "activate_workspace", AsyncMock(return_value=_fake_ws())),
        patch.object(agents_mod, "stage_active_workspaces_for_user", stage),
        patch.object(agents_mod, "finalize_archived_workspaces", finalize),
        patch.object(agents_mod, "pick_agent_avatar", MagicMock(return_value=MagicMock(key="k", url="u"))),
        patch.object(agents_mod, "set_progress", AsyncMock()),
        patch.object(agents_mod, "_enqueue_agent_initialization", AsyncMock()),
    )


@pytest.mark.asyncio
async def test_template_path_does_not_archive_siblings():
    stage = AsyncMock()
    finalize = AsyncMock()
    fake_aiagent = MagicMock()
    fake_aiagent.find_first = AsyncMock()  # pending guard — must NOT be called
    fake_aiagent.create = AsyncMock(return_value=_FakeAgent())
    fake_aiagent.update = AsyncMock()

    with __import__("contextlib").ExitStack() as stack:
        for p in _provisioning_patches(fake_aiagent, stage, finalize):
            stack.enter_context(p)
        agent, _ws = await agents_mod.create_agent_with_provisioning(
            user_id="sys-owner", name="小伴", personality={},
            stage_existing_workspaces=False,
        )

    assert agent.id == "tpl-new"
    stage.assert_not_called()  # ← no sibling template archiving
    finalize.assert_not_called()
    fake_aiagent.find_first.assert_not_called()  # ← pending-409 guard skipped


@pytest.mark.asyncio
async def test_normal_path_still_stages_and_guards():
    stage = AsyncMock(return_value=[])
    finalize = AsyncMock()
    fake_aiagent = MagicMock()
    fake_aiagent.find_first = AsyncMock(return_value=None)
    fake_aiagent.create = AsyncMock(return_value=_FakeAgent("agent-1"))
    fake_aiagent.update = AsyncMock()

    with __import__("contextlib").ExitStack() as stack:
        for p in _provisioning_patches(fake_aiagent, stage, finalize):
            stack.enter_context(p)
        await agents_mod.create_agent_with_provisioning(
            user_id="u1", name="A", personality={},
        )

    stage.assert_awaited_once()  # normal user: still archives previous companion
    fake_aiagent.find_first.assert_awaited_once()  # pending guard runs


@pytest.mark.asyncio
async def test_list_template_agents_includes_archived():
    captured: dict = {}

    async def _find_many(where=None, order=None):
        captured["where"] = where
        return []

    fake_aiagent = MagicMock()
    fake_aiagent.find_many = _find_many
    fake_user = MagicMock()
    fake_user.find_unique = AsyncMock(return_value=type("U", (), {"id": "sys-owner"})())

    with patch.object(reg.db, "aiagent", fake_aiagent), \
         patch.object(reg.db, "user", fake_user):
        await reg.list_template_agents()

    # No status filter → archived (legacy) templates are still listed & deletable.
    assert captured["where"] == {"userId": "sys-owner"}
