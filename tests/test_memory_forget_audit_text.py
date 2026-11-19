"""遗忘审计与日志不留记忆正文 (M02.04.05)。"""
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services.memory.interaction import deletion

SECRET = "用户养了一只叫团子的猫"


def _rec(mid="m1"):
    return SimpleNamespace(id=mid, userId="u1", workspaceId="w1", source="user", content=SECRET,
                           type="life", mainCategory="生活", subCategory="宠物", level=2,
                           importance=0.6, isArchived=False, occurTime=None, statementTime=None,
                           recurrence=None, provenance=None)


@pytest.mark.asyncio
async def test_describe_delete_audits_without_text(caplog):
    caplog.set_level(logging.INFO, logger=deletion.logger.name)
    with patch.object(deletion, "generate_embedding", AsyncMock(return_value=[0.1])), \
         patch.object(deletion, "search_by_embedding",
                      AsyncMock(return_value=[{"id": "m1", "similarity": 0.99, "content": SECRET}])), \
         patch.object(deletion.memory_repo, "find_unique", AsyncMock(return_value=_rec())), \
         patch.object(deletion.memory_repo, "delete", AsyncMock()), \
         patch.object(deletion, "log_memory_changelog", AsyncMock()) as audit:
        assert await deletion.delete_memories_by_description("u1", "猫", "w1") == 1
    assert audit.await_args.args == ("u1", "m1", "delete")
    assert not audit.await_args.kwargs
    assert SECRET not in caplog.text and "团子" not in caplog.text


@pytest.mark.asyncio
async def test_confirmed_delete_audits_without_text_but_keeps_undo(caplog):
    caplog.set_level(logging.INFO, logger=deletion.logger.name)
    with patch.object(deletion.memory_repo, "find_unique", AsyncMock(return_value=_rec())), \
         patch.object(deletion.memory_repo, "delete", AsyncMock()), \
         patch.object(deletion, "save_delete_undo", AsyncMock()) as undo, \
         patch.object(deletion, "log_memory_changelog", AsyncMock()) as audit:
        assert await deletion.execute_confirmed_deletion(
            "u1", [{"id": "m1"}], conversation_id="c1") == 1
    assert audit.await_args.args == ("u1", "m1", "delete") and not audit.await_args.kwargs
    assert undo.await_args.kwargs["snapshots"][0]["content"] == SECRET
    assert SECRET not in caplog.text


@pytest.mark.asyncio
async def test_undo_logs_ids_not_text(caplog):
    caplog.set_level(logging.INFO, logger=deletion.logger.name)
    snap = deletion._snapshot_memory(_rec())
    with patch("app.services.memory.storage.persistence.store_memory",
               AsyncMock(side_effect=["new-id-1", RuntimeError(SECRET)])):
        assert await deletion.restore_deleted_memories([snap, snap]) == 1
    assert SECRET not in caplog.text and "new-id-1"[:8] in caplog.text


def test_memory_write_and_recall_logs_carry_no_text():
    """记忆写入与召回路径的普通日志只记 ID / 长度, 不截取正文或用户消息。"""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "app"
    files = [*sorted((root / "services" / "memory").rglob("*.py")), root / "services" / "chat" / "data_fetch_phase.py"]
    pattern = re.compile(
        r"logger\.\w+\(.*\b(content|text|summary|message|query|cache_key)\b(\.get\([^)]*\))?\s*\[:\d+\]")
    offenders = [f"{p.relative_to(root)}:{i}" for p in files
                 for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
                 if pattern.search(line) or re.search(r"logger\.\w+\(.*get\('content'[^)]*\)\[:", line)]
    assert offenders == []
