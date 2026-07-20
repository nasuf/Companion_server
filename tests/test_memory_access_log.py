from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_access_log_batches_changelog_and_touches_snake_case():
    from app.services.memory.retrieval.access_log import log_memory_access

    execute_raw = AsyncMock()
    with patch("app.services.memory.retrieval.access_log.db") as mock_db:
        mock_db.execute_raw = execute_raw
        await log_memory_access("u1", ["m1", "m2", "m3"], workspace_id="ws1")

    sql_calls = [call.args[0] for call in execute_raw.await_args_list]
    # 1 batched changelog INSERT + 2 updated_at touches (user + ai tables).
    assert len(sql_calls) == 3
    insert_sql = sql_calls[0]
    assert "INSERT INTO memory_changelogs" in insert_sql
    assert insert_sql.count("'access'") == 3  # one VALUES tuple per memory
    # Batched insert carries (id, user_id, workspace_id, memory_id) per row.
    insert_args = execute_raw.await_args_list[0].args[1:]
    assert len(insert_args) == 12
    assert insert_args[1] == "u1" and insert_args[3] == "m1"
    # Touch statements use actual DB column names.
    for sql in sql_calls[1:]:
        assert "updated_at" in sql
        assert "updatedAt" not in sql


@pytest.mark.asyncio
async def test_access_log_empty_ids_noop():
    from app.services.memory.retrieval.access_log import log_memory_access

    with patch("app.services.memory.retrieval.access_log.db") as mock_db:
        mock_db.execute_raw = AsyncMock()
        await log_memory_access("u1", [], workspace_id="ws1")
        mock_db.execute_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_access_log_insert_failure_still_touches_timestamps():
    """Changelog write failing must not abort the updatedAt touch (best-effort)."""
    from app.services.memory.retrieval.access_log import log_memory_access

    calls: list[str] = []

    async def _execute(sql, *args):
        calls.append(sql)
        if "INSERT" in sql:
            raise RuntimeError("db hiccup")

    with patch("app.services.memory.retrieval.access_log.db") as mock_db:
        mock_db.execute_raw = AsyncMock(side_effect=_execute)
        await log_memory_access("u1", ["m1"], workspace_id="ws1")

    assert sum("UPDATE" in c for c in calls) == 2
