"""合并不能造出"存了也检索不到"的记忆.

`memory.reconciliation` 让 LLM 把两条记忆合成一条, 但 prompt 里没有任何长度约束 ——
拼接式的 merged_summary 很容易超过检索单条上限, 而超限的记忆会被 context_selector
整条跳过。那样等于**拿两条能用的换一条用不了的**, 是净损失。

2026-07 刚修完 606 条这种死记忆 (还把长记忆按句子拆成了多条), 如果不堵这个入口,
hygiene 下一次跑就可能把拆开的兄弟片段合回去, 修复直接被撤销。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.memory.retrieval.context_selector import (
    MAX_MEMORY_TOKENS_PER_ITEM,
    estimate_tokens,
)
from app.services.memory.storage import reconciliation as rc

_LONG = "这是一段被合并后变得很长的记忆内容需要占据足够多的字数才能越过上限。" * 8
_SHORT = "我养了一只叫大橘的猫。"


def _fixture_args() -> dict:
    return dict(
        source="ai",
        old_main="生活", old_sub="宠物", old_text="我养了一只猫。",
        new_main="生活", new_sub="宠物", new_text=_SHORT,
    )


async def _adjudicate(llm_result: dict):
    # 这两个依赖是在函数体内 import 的, 必须在**源模块**上打桩;
    # 在 reconciliation 模块上 patch 只会新建一个没人读的属性, 请求会真打到 LLM。
    with patch(
        "app.services.prompting.store.get_prompt_text",
        AsyncMock(return_value="{old_text}{new_text}"),
    ), patch(
        "app.services.llm.models.invoke_json", AsyncMock(return_value=llm_result)
    ), patch("app.services.llm.models.get_chat_model", lambda *a, **k: object()):
        return await rc._llm_adjudicate(**_fixture_args())


class TestGuard:
    def test_the_fixture_is_actually_oversized(self):
        """样本本身不超限的话, 下面的测试就什么都没测到."""
        assert estimate_tokens(_LONG) > MAX_MEMORY_TOKENS_PER_ITEM
        assert estimate_tokens(_SHORT) <= MAX_MEMORY_TOKENS_PER_ITEM

    def test_helper_matches_the_selector_limit(self):
        assert rc._exceeds_injection_limit(_LONG) is True
        assert rc._exceeds_injection_limit(_SHORT) is False

    def test_empty_text_is_not_flagged(self):
        assert rc._exceeds_injection_limit("") is False


@pytest.mark.asyncio
class TestAdjudication:
    async def test_oversized_merge_is_downgraded_to_keep_separate(self):
        d = await _adjudicate({"action": "merge_existing", "merged_summary": _LONG})
        assert d is not None
        assert d.action == "keep_separate"
        assert d.merged_content is None
        assert d.reason == "merged_too_long"

    async def test_oversized_update_is_also_blocked(self):
        """update_existing 会用新文本覆盖老行 —— 覆盖成一条死记忆同样是净损失."""
        d = await _adjudicate({"action": "update_existing", "merged_summary": _LONG})
        assert d is not None
        assert d.action == "keep_separate"

    async def test_normal_merge_still_goes_through(self):
        """护栏不能把正常合并也拦掉."""
        d = await _adjudicate({"action": "merge_existing", "merged_summary": _SHORT})
        assert d is not None
        assert d.action == "merge_existing"
        assert d.merged_content == _SHORT

    async def test_non_merge_actions_are_untouched(self):
        d = await _adjudicate({"action": "keep_separate", "reason": "不同的事"})
        assert d is not None
        assert d.action == "keep_separate"
        assert d.reason == "不同的事"
