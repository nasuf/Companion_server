"""Phase 0.5: workspace_id 严格过滤 — 防 cross-workspace leak.

历史 SQL: ($2::text IS NULL OR workspace_id = $2) — workspace_id=None 时
跨 workspace 全查. 用户产品 1:1 (1 user 1 agent 1 workspace), workspace_id
不该 None. 改为严格过滤: None → 返空 + warning, 防数据漂移泄露.

保留 IS NULL 模式的: archive_stale_entities (admin batch), _load_active_entities
(entity merge admin). 这些是 cron 路径, intentional cross-workspace scan.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_top_entities_workspace_none_returns_empty():
    """Phase 0.5: workspace_id=None → 返空 + warning, 不跨 workspace 全查."""
    from app.services.memory.storage import entity_repo

    # 即使 db 不被调, 也要确认逻辑路径
    with patch.object(entity_repo, "logger") as mock_logger:
        result = await entity_repo.top_entities(
            user_id="u1", workspace_id=None,
        )

    assert result == []
    # warning 必须 emit (admin 可监控)
    assert mock_logger.warning.called
    msg = mock_logger.warning.call_args.args[0]
    assert "workspace_id=None" in msg
    assert "u1" in msg


@pytest.mark.asyncio
async def test_get_user_preferences_workspace_none_returns_empty():
    from app.services.memory.storage import entity_repo

    with patch.object(entity_repo, "logger"):
        result = await entity_repo.get_user_preferences(
            user_id="u1", workspace_id=None,
        )

    assert result == []


@pytest.mark.asyncio
async def test_get_related_memories_workspace_none_returns_empty():
    from app.services.memory.storage import entity_repo

    with patch.object(entity_repo, "logger"):
        result = await entity_repo.get_related_memories(
            user_id="u1", workspace_id=None, entity_name="X",
        )

    assert result == []


@pytest.mark.asyncio
async def test_get_relationship_context_workspace_none_categories_empty():
    """get_relationship_context: workspace_id=None → categories 段空,
    topics/entities 段也因 top_entities 返空而空, 整体 graceful degrade."""
    from app.services.memory.storage import entity_repo

    with patch.object(entity_repo, "logger"):
        result = await entity_repo.get_relationship_context(
            user_id="u1", workspace_id=None,
        )

    assert result["topics"] == []
    assert result["entities"] == []
    assert result["categories"] == []


@pytest.mark.asyncio
async def test_top_entities_with_valid_workspace_uses_strict_filter():
    """workspace_id 有值时, SQL 走严格 = 过滤 (不再 IS NULL OR 模式)."""
    from app.services.memory.storage import entity_repo

    captured_sql = []
    async def _capture(sql, *args):
        captured_sql.append(sql)
        return []

    with patch.object(entity_repo, "db") as mock_db:
        mock_db.query_raw = AsyncMock(side_effect=_capture)
        await entity_repo.top_entities(
            user_id="u1", workspace_id="ws1",
        )

    assert len(captured_sql) == 1
    sql = captured_sql[0]
    # 不再有 IS NULL OR 模式
    assert "IS NULL OR workspace_id" not in sql
    # 必须有严格 workspace_id = $X 过滤
    assert "workspace_id = $2" in sql


@pytest.mark.asyncio
async def test_archive_stale_entities_keeps_isnull_pattern_admin():
    """admin batch (archive_stale_entities) intentional cross-workspace scan,
    必须保留 IS NULL OR 模式 (workspace_id=None 时全库扫). 不该被 Phase 0.5
    错误改造."""
    import inspect
    from app.services.memory.storage import entity_repo

    src = inspect.getsource(entity_repo.archive_stale_entities)
    assert "IS NULL OR" in src, (
        "archive_stale_entities 是 admin cron, 必须保留 IS NULL OR 模式 "
        "支持全库扫"
    )
