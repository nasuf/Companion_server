"""Creating a new template must not archive the template user's other templates.

The template system user owns MANY coexisting templates (one is the default).
create_agent_with_provisioning was built for the normal single-companion model,
so it staged (archived) the user's other active workspaces — which archived
every previously-created template, hiding them from the admin list and making
them undeletable. The template path now passes stage_existing_workspaces=False
AND opts the new workspace out of chat_workspaces_user_id_active_key
(allow_multiple_active=True); otherwise activate_workspace hits UniqueViolation
on user_id as soon as a second template is created.
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


def _provisioning_patches(fake_aiagent, stage, finalize, *, create_ws=None):
    if create_ws is None:
        create_ws = AsyncMock(return_value=_fake_ws())
    return (
        patch.object(agents_mod.db, "aiagent", fake_aiagent),
        patch.object(agents_mod, "create_provisioning_workspace", create_ws),
        patch.object(agents_mod, "activate_workspace", AsyncMock(return_value=_fake_ws())),
        patch.object(agents_mod, "stage_active_workspaces_for_user", stage),
        patch.object(agents_mod, "finalize_archived_workspaces", finalize),
        patch.object(agents_mod, "pick_agent_avatar", MagicMock(return_value=MagicMock(key="k", url="u"))),
        patch.object(agents_mod, "set_progress", AsyncMock()),
        patch.object(agents_mod, "_enqueue_agent_initialization", AsyncMock()),
    ), create_ws


@pytest.mark.asyncio
async def test_template_path_does_not_archive_siblings():
    stage = AsyncMock()
    finalize = AsyncMock()
    fake_aiagent = MagicMock()
    fake_aiagent.find_first = AsyncMock()  # pending guard — must NOT be called
    fake_aiagent.create = AsyncMock(return_value=_FakeAgent())
    fake_aiagent.update = AsyncMock()

    with __import__("contextlib").ExitStack() as stack:
        patches, create_ws = _provisioning_patches(fake_aiagent, stage, finalize)
        for p in patches:
            stack.enter_context(p)
        agent, _ws = await agents_mod.create_agent_with_provisioning(
            user_id="sys-owner", name="小伴", personality={},
            stage_existing_workspaces=False,
        )

    assert agent.id == "tpl-new"
    stage.assert_not_called()  # ← no sibling template archiving
    finalize.assert_not_called()
    fake_aiagent.find_first.assert_not_called()  # ← pending-409 guard skipped
    create_ws.assert_awaited_once_with(
        "sys-owner", "tpl-new", allow_multiple_active=True,
    )


@pytest.mark.asyncio
async def test_normal_path_still_stages_and_guards():
    stage = AsyncMock(return_value=[])
    finalize = AsyncMock()
    fake_aiagent = MagicMock()
    fake_aiagent.find_first = AsyncMock(return_value=None)
    fake_aiagent.create = AsyncMock(return_value=_FakeAgent("agent-1"))
    fake_aiagent.update = AsyncMock()

    with __import__("contextlib").ExitStack() as stack:
        patches, create_ws = _provisioning_patches(fake_aiagent, stage, finalize)
        for p in patches:
            stack.enter_context(p)
        await agents_mod.create_agent_with_provisioning(
            user_id="u1", name="A", personality={},
        )

    stage.assert_awaited_once()  # normal user: still archives previous companion
    fake_aiagent.find_first.assert_awaited_once()  # pending guard runs
    create_ws.assert_awaited_once_with(
        "u1", "agent-1", allow_multiple_active=False,
    )


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


@pytest.mark.asyncio
async def test_template_unique_violation_becomes_409():
    """If the unique index still fires (migration not applied), return 409 not 500."""
    from fastapi import HTTPException
    from prisma.errors import UniqueViolationError

    stage = AsyncMock()
    finalize = AsyncMock()
    fake_aiagent = MagicMock()
    fake_aiagent.find_first = AsyncMock()
    fake_aiagent.create = AsyncMock(return_value=_FakeAgent())
    fake_aiagent.update = AsyncMock()
    uv = UniqueViolationError({
        "user_facing_error": {
            "message": "Unique constraint failed on the fields: (`user_id`)",
        }
    })

    with __import__("contextlib").ExitStack() as stack:
        patches, _create_ws = _provisioning_patches(fake_aiagent, stage, finalize)
        for p in patches:
            stack.enter_context(p)
        stack.enter_context(patch.object(
            agents_mod, "activate_workspace", AsyncMock(side_effect=uv),
        ))
        stack.enter_context(patch.object(
            agents_mod, "archive_provisioning_workspace", AsyncMock(),
        ))
        with pytest.raises(HTTPException) as ei:
            await agents_mod.create_agent_with_provisioning(
                user_id="sys-owner", name="小伴", personality={},
                stage_existing_workspaces=False,
            )

    assert ei.value.status_code == 409
    assert "模板" in str(ei.value.detail)
    fake_aiagent.update.assert_awaited()  # rolled back to archived


@pytest.mark.asyncio
async def test_provisioning_workspace_sets_flag_via_raw_sql():
    from app.services.workspace import workspaces as ws_mod

    created = _fake_ws()
    create = AsyncMock(return_value=created)
    execute_raw = AsyncMock()
    fake_db = MagicMock()
    fake_db.chatworkspace.create = create
    fake_db.execute_raw = execute_raw

    with patch.object(ws_mod, "db", fake_db):
        result = await ws_mod.create_provisioning_workspace(
            "sys-owner", "tpl-new", allow_multiple_active=True,
        )

    assert result is created
    assert create.await_args.kwargs["data"]["status"] == "provisioning"
    execute_raw.assert_awaited_once()
    sql, ws_id = execute_raw.await_args.args
    assert "allow_multiple_active = TRUE" in sql
    assert ws_id == created.id


@pytest.mark.asyncio
async def test_provisioning_workspace_archives_if_flag_write_fails():
    from app.services.workspace import workspaces as ws_mod

    created = _fake_ws()
    fake_db = MagicMock()
    fake_db.chatworkspace.create = AsyncMock(return_value=created)
    fake_db.chatworkspace.update = AsyncMock()
    fake_db.execute_raw = AsyncMock(side_effect=RuntimeError("no column"))

    with patch.object(ws_mod, "db", fake_db):
        with pytest.raises(RuntimeError, match="no column"):
            await ws_mod.create_provisioning_workspace(
                "sys-owner", "tpl-new", allow_multiple_active=True,
            )

    fake_db.chatworkspace.update.assert_awaited_once()
    assert fake_db.chatworkspace.update.await_args.kwargs["where"]["id"] == created.id
    assert fake_db.chatworkspace.update.await_args.kwargs["data"]["status"] == "archived"


@pytest.mark.asyncio
async def test_provisioning_workspace_skips_flag_for_regular_users():
    from app.services.workspace import workspaces as ws_mod

    fake_db = MagicMock()
    fake_db.chatworkspace.create = AsyncMock(return_value=_fake_ws())
    fake_db.execute_raw = AsyncMock()

    with patch.object(ws_mod, "db", fake_db):
        await ws_mod.create_provisioning_workspace("u1", "agent-1")

    fake_db.execute_raw.assert_not_called()


def test_unique_index_migration_exempts_shared_owner():
    from pathlib import Path

    sql = (
        Path(__file__).resolve().parents[1]
        / "prisma/migrations/20260912140000_template_multi_active_workspace/migration.sql"
    ).read_text()
    assert "allow_multiple_active" in sql
    assert 'WHERE "status" = \'active\' AND "allow_multiple_active" = FALSE' in sql
    assert "__companion_template_system__" in sql
