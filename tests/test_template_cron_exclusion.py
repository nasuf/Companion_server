"""The template agent must be excluded from every per-agent / per-workspace cron.

Root cause (2026-07 production audit): the template agent (owned by the template
system user) is a fully-provisioned, `active` AiAgent. The daily crons enumerated
*all* agents with an unfiltered `find_many()`, so the template ran daily schedule
generation + daily self-memory summaries + proactive sends on itself. Those
self-memories then got copied into every new clone, so later clones inherited a
growing pile of junk "daily life" memories. The template must stay a frozen
clone source: excluded from all runtime crons.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.agent_template import registry as reg
from app.services.proactive import state as state_mod
from app.services.proactive import special_dates as sd_mod
from jobs import scheduler as sched


@pytest.fixture(autouse=True)
def _reset_owner_cache():
    reg._template_owner_id_cache = None
    yield
    reg._template_owner_id_cache = None


# ── registry.get_template_owner_id ────────────────────────────────────────

def _fake_user_actions(find_return):
    # Prisma action objects expose read-only methods, so swap the whole
    # ``db.user`` namespace with a mock instead of patching its attributes.
    fake = MagicMock()
    fake.find_unique = AsyncMock(return_value=find_return)
    fake.create = AsyncMock()
    return fake


@pytest.mark.asyncio
async def test_get_template_owner_id_readonly_and_cached():
    fake_user = _fake_user_actions(type("U", (), {"id": "sys-owner"})())
    with patch.object(reg.db, "user", fake_user):
        first = await reg.get_template_owner_id()
        second = await reg.get_template_owner_id()

    assert first == "sys-owner"
    assert second == "sys-owner"
    # Read-only: never creates the user, and the cache means only one lookup.
    fake_user.create.assert_not_called()
    assert fake_user.find_unique.await_count == 1


@pytest.mark.asyncio
async def test_get_template_owner_id_none_when_absent():
    fake_user = _fake_user_actions(None)
    with patch.object(reg.db, "user", fake_user):
        assert await reg.get_template_owner_id() is None


# ── jobs.scheduler._run_for_all_agents ────────────────────────────────────

def _fake_aiagent_actions(captured):
    async def _find_many(where=None):
        captured["where"] = where
        return []  # empty → no per-agent processing / infra needed
    fake = MagicMock()
    fake.find_many = _find_many
    return fake


@pytest.mark.asyncio
async def test_run_for_all_agents_excludes_template_and_archived():
    from app.db import db
    captured: dict = {}
    with patch.object(reg, "get_template_owner_id", AsyncMock(return_value="sys-owner")), \
         patch.object(db, "aiagent", _fake_aiagent_actions(captured)):
        await sched._run_for_all_agents(AsyncMock(), concurrency=1, task_name="t")

    assert captured["where"] == {"status": "active", "userId": {"not": "sys-owner"}}


@pytest.mark.asyncio
async def test_run_for_all_agents_filters_active_even_without_template_user():
    from app.db import db
    captured: dict = {}
    with patch.object(reg, "get_template_owner_id", AsyncMock(return_value=None)), \
         patch.object(db, "aiagent", _fake_aiagent_actions(captured)):
        await sched._run_for_all_agents(AsyncMock(), concurrency=1, task_name="t")

    # Still skips archived/provisioning agents even when no template exists.
    assert captured["where"] == {"status": "active"}


# ── proactive.state list functions ────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_due_states_excludes_template_owner():
    captured: dict = {}

    async def _fake_query_raw(sql, *args):
        captured["sql"] = sql
        captured["args"] = args
        return []

    with patch.object(reg, "get_template_owner_id", AsyncMock(return_value="sys-owner")), \
         patch.object(state_mod.db, "query_raw", _fake_query_raw):
        await state_mod.list_due_proactive_states()

    assert "user_id <> $2" in captured["sql"]
    assert captured["args"][1] == "sys-owner"


@pytest.mark.asyncio
async def test_list_waiting_states_excludes_template_owner():
    captured: dict = {}

    async def _fake_query_raw(sql, *args):
        captured["sql"] = sql
        captured["args"] = args
        return []

    with patch.object(reg, "get_template_owner_id", AsyncMock(return_value="sys-owner")), \
         patch.object(state_mod.db, "query_raw", _fake_query_raw):
        await state_mod.list_waiting_timeout_states()

    assert "user_id <> $2" in captured["sql"]
    assert captured["args"][1] == "sys-owner"


@pytest.mark.asyncio
async def test_list_due_states_no_template_passes_null_guard():
    """No template user → owner_id None; the SQL guard `$2 IS NULL` keeps all rows."""
    captured: dict = {}

    async def _fake_query_raw(sql, *args):
        captured["args"] = args
        return []

    with patch.object(reg, "get_template_owner_id", AsyncMock(return_value=None)), \
         patch.object(state_mod.db, "query_raw", _fake_query_raw):
        await state_mod.list_due_proactive_states()

    assert captured["args"][1] is None


# ── proactive.special_dates.scan_special_dates_today ──────────────────────

@pytest.mark.asyncio
async def test_scan_special_dates_excludes_template():
    captured: dict = {}
    with patch.object(reg, "get_template_owner_id", AsyncMock(return_value="sys-owner")), \
         patch.object(sd_mod.db, "aiagent", _fake_aiagent_actions(captured)):
        await sd_mod.scan_special_dates_today()

    assert captured["where"] == {"status": "active", "userId": {"not": "sys-owner"}}
