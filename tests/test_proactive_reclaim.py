"""Stale `processing` reclaim for proactive states.

A state claimed into `processing`/`processing_timeout` whose owning instance
crashed mid-send would stall that workspace forever. reclaim_stale_processing_states
resets aged states to their re-eligible predecessor status.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app.services.proactive import state as state_mod


@pytest.mark.asyncio
async def test_reclaim_issues_guarded_update():
    """SQL 只重置 processing*/超时的行, 并映射回可再调度状态."""
    captured = {}

    async def _fake_query_raw(sql, *args):
        captured["sql"] = sql
        captured["args"] = args
        return [{"id": "s1", "workspace_id": "w1"}]

    with patch.object(state_mod.db, "query_raw", _fake_query_raw):
        n = await state_mod.reclaim_stale_processing_states(
            now=datetime(2026, 7, 5, tzinfo=timezone.utc), timeout_s=300,
        )

    assert n == 1
    sql = captured["sql"]
    assert "status IN ('processing', 'processing_timeout')" in sql
    assert "WHEN 'processing' THEN 'running'" in sql
    assert "WHEN 'processing_timeout' THEN 'waiting_user'" in sql
    assert "last_attempt_at < $1::timestamp" in sql


@pytest.mark.asyncio
async def test_reclaim_returns_zero_when_none_stale():
    with patch.object(state_mod.db, "query_raw", AsyncMock(return_value=[])):
        n = await state_mod.reclaim_stale_processing_states(timeout_s=300)
    assert n == 0


@pytest.mark.asyncio
async def test_reclaim_swallows_db_errors():
    with patch.object(state_mod.db, "query_raw", AsyncMock(side_effect=Exception("db down"))):
        n = await state_mod.reclaim_stale_processing_states(timeout_s=300)
    assert n == 0


@pytest.mark.asyncio
async def test_scan_calls_reclaim_first():
    """orchestrator.scan_proactive_states 先回收僵死态再列出到期态."""
    from app.services.proactive import orchestrator as orch

    order: list[str] = []

    async def _fake_reclaim(*, now=None):
        order.append("reclaim")
        return 0

    async def _fake_list_due(now=None):
        order.append("list_due")
        return []

    async def _fake_list_waiting(now=None):
        order.append("list_waiting")
        return []

    with (
        patch.object(orch, "reclaim_stale_processing_states", _fake_reclaim),
        patch.object(orch, "list_due_proactive_states", _fake_list_due),
        patch.object(orch, "list_waiting_timeout_states", _fake_list_waiting),
    ):
        await orch.scan_proactive_states()

    assert order[0] == "reclaim"
    assert "list_due" in order
