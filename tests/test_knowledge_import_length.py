"""后台 txt 导入的知识条目不能超过检索注入上限.

超过就会在检索时被整条跳过 —— 存进去了, 但**任何对话都不会用到它**。而管理员只会
看到"导入成功 N 条", 完全不知道其中几条是死的。2026-07 实测库里 5% 的知识条目处于
这个状态。

刻意不自动截断也不自动拆分: txt 是管理员给的既成内容, 系统擅自改写会让"我上传的"
和"库里存的"对不上。正确做法是挑出来告诉他, 由他决定怎么拆。
"""

from __future__ import annotations

import pytest

from app.services.agent_template.knowledge_import import (
    MAX_KNOWLEDGE_ITEM_TOKENS,
    KnowledgeItem,
    oversized_items,
)
from app.services.memory.retrieval.context_selector import (
    MAX_MEMORY_TOKENS_PER_ITEM,
    estimate_tokens,
)


def _item(summary: str) -> KnowledgeItem:
    return KnowledgeItem(section="测试", label="标签", content=summary, summary=summary)


class TestThresholdAlignment:
    def test_import_limit_matches_the_injection_limit(self):
        """两个数必须一致.

        导入放得比注入宽 → 存进去检索不到 (就是现在的 bug);
        导入卡得比注入严 → 白白拒绝本来能用的内容。
        """
        assert MAX_KNOWLEDGE_ITEM_TOKENS == MAX_MEMORY_TOKENS_PER_ITEM


class TestDetection:
    def test_normal_item_is_not_flagged(self):
        item = _item("西甲联赛的赛事时间：每年 8 月到次年 5 月")
        assert item.is_oversized is False

    def test_long_item_is_flagged(self):
        long_text = "这是一条很长的知识内容用来测试长度限制是否生效。" * 12
        assert estimate_tokens(long_text) > MAX_KNOWLEDGE_ITEM_TOKENS
        assert _item(long_text).is_oversized is True

    def test_oversized_items_filters_the_batch(self):
        items = [
            _item("短条目一"),
            _item("很长的内容。" * 60),
            _item("短条目二"),
        ]
        flagged = oversized_items(items)
        assert len(flagged) == 1
        assert flagged[0].summary.startswith("很长的内容")

    def test_empty_batch_safe(self):
        assert oversized_items([]) == []


class TestStoragePath:
    @pytest.mark.asyncio
    async def test_oversized_items_are_not_stored_and_are_reported(self, monkeypatch):
        """超长条目不入库, 且原文回给管理员 —— 只给个计数他无从下手."""
        from unittest.mock import AsyncMock

        import app.services.agent_template.knowledge as kn

        workspace = type("W", (), {"id": "ws-1"})()
        monkeypatch.setattr(kn, "get_active_workspace", AsyncMock(return_value=workspace))
        monkeypatch.setattr(kn, "_knowledge_contents", AsyncMock(return_value=set()))
        monkeypatch.setattr(kn, "_bust_knowledge_rows_cache", AsyncMock())

        stored_texts: list[str] = []

        async def _store(_user, text, **_kw):
            stored_texts.append(text)
            return "mem-id"

        monkeypatch.setattr(kn, "store_memory", _store)

        # 无锁环境: distributed_lock 在没有 Redis 时 fail_open, 这里直接放行。
        class _NoLock:
            async def __aenter__(self):
                return True

            async def __aexit__(self, *a):
                return False

        monkeypatch.setattr(kn, "distributed_lock", lambda *a, **k: _NoLock())

        long_text = "超长知识内容需要被挡下来。" * 30
        result = await kn.append_knowledge_to_template(
            template_agent_id="agent-1",
            template_user_id="user-1",
            items=[_item("正常的一条知识"), _item(long_text)],
        )

        assert stored_texts == ["正常的一条知识"], "超长条目被存进去了"
        assert result["stored"] == 1
        assert len(result["skipped_oversized"]) == 1
        assert result["skipped_oversized"][0].startswith("超长知识内容"), (
            "只回了计数没回原文, 管理员不知道是哪条"
        )

    @pytest.mark.asyncio
    async def test_report_is_present_even_when_nothing_is_oversized(self, monkeypatch):
        """字段恒在, 前端不必判 key 是否存在."""
        from unittest.mock import AsyncMock

        import app.services.agent_template.knowledge as kn

        monkeypatch.setattr(
            kn, "get_active_workspace",
            AsyncMock(return_value=type("W", (), {"id": "ws-1"})()),
        )
        monkeypatch.setattr(kn, "_knowledge_contents", AsyncMock(return_value=set()))
        monkeypatch.setattr(kn, "_bust_knowledge_rows_cache", AsyncMock())
        monkeypatch.setattr(kn, "store_memory", AsyncMock(return_value="mem"))

        class _NoLock:
            async def __aenter__(self):
                return True

            async def __aexit__(self, *a):
                return False

        monkeypatch.setattr(kn, "distributed_lock", lambda *a, **k: _NoLock())

        result = await kn.append_knowledge_to_template(
            template_agent_id="a", template_user_id="u", items=[_item("短的")],
        )
        assert result["skipped_oversized"] == []
