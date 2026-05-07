from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_access_log_touches_snake_case_updated_at_column():
    from app.services.memory.retrieval.access_log import log_memory_access

    execute_raw = AsyncMock()
    with patch("app.services.memory.retrieval.access_log.log_memory_changelog", AsyncMock()), \
         patch("app.services.memory.retrieval.access_log.db") as mock_db:
        mock_db.execute_raw = execute_raw

        await log_memory_access("u1", ["m1"], workspace_id="ws1")

    sql_calls = [call.args[0] for call in execute_raw.await_args_list]
    assert len(sql_calls) == 2
    assert all("updated_at" in sql for sql in sql_calls)
    assert all("updatedAt" not in sql for sql in sql_calls)
