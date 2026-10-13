"""Keep the historical contradiction policy checks on the actual atomic writer."""

from unittest.mock import AsyncMock
import pytest

from app.services.memory.interaction import (
    contradiction as c,
    contradiction_write as writer,
)
from app.services.memory.manual_write import MemoryWriteError
from tests.test_memory_evidence_postgres import origins
from tests.test_contradiction_write_postgres import confirming


@pytest.mark.parametrize(
    "kind,old,expected",
    [
        ("变化", 0.95, 0.9),
        ("变化", 0.86, 0.85),
        ("错误", 0.99, 0.7),
        ("新增", 0.9, 0.85),
    ],
)
def test_importance_policy(kind, old, expected):
    assert c._resolve_new_importance(kind, old) == pytest.approx(expected)


@pytest.mark.parametrize(
    "main,sub,content",
    [("身份", "姓名", "用户叫馒头"), ("身份", "现居地", "用户住在乙城")],
)
async def test_updated_memory_replacement_uses_explicit_categories(
    confirming, main, sub, content
):
    f = confirming
    result = await f.apply(
        analysis={
            "change_type": "变化",
            "updated_memory": content,
            "new_memory": "",
            "new_memory_main_category": main,
            "new_memory_sub_category": sub,
        }
    )
    row = await f.db.usermemory.find_unique(where={"id": result["memory_id"]})
    assert row.content == content and (row.mainCategory, row.subCategory) == (main, sub)
    assert (await f.db.usermemory.find_unique(where={"id": f.old.id})).isArchived


async def test_fallback_categories_are_preserved(confirming):
    f = confirming
    result = await f.apply()
    row = await f.db.usermemory.find_unique(where={"id": result["memory_id"]})
    assert (
        row.mainCategory == f.old.mainCategory and row.subCategory == f.old.subCategory
    )


async def test_missing_old_memory_is_not_silently_successful(confirming):
    f = confirming
    await f.db.usermemory.delete(where={"id": f.old.id})
    with pytest.raises(MemoryWriteError, match="memory_changed"):
        await f.apply()
    f.embedding.assert_not_awaited()


@pytest.mark.parametrize(
    "analysis",
    [
        {"change_type": "bogus", "new_memory": "信息"},
        {"change_type": "变化", "new_memory": ["wrong"]},
        {"change_type": "变化", "new_memory": "x" * 10001},
    ],
)
async def test_invalid_model_result_cannot_mutate(confirming, analysis):
    f = confirming
    with pytest.raises(MemoryWriteError, match="analysis_invalid"):
        await f.apply(analysis=analysis)
    assert not (await f.db.usermemory.find_unique(where={"id": f.old.id})).isArchived


async def test_public_apply_delegates_result_and_required_scope(monkeypatch):
    apply = AsyncMock(return_value={"status": "applied", "memory_id": "new"})
    monkeypatch.setattr(writer, "apply_confirmed_change", apply)
    result = await c.apply_contradiction_resolution(
        {"old": "x"}, {"change": "y"}, user_id="owner"
    )
    assert result["memory_id"] == "new"
    apply.assert_awaited_once_with({"old": "x"}, {"change": "y"}, user_id="owner")
