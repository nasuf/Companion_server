"""线下活动「外出情境」段注入回归。

保证: 进行中活动注入 → 回复贴合外出; 无活动 → 不注入; 且**永不泄露**拍摄物品/任务目标。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.chat.prompt_builder import _build_offline_activity_section

PB = "app.services.chat.prompt_builder"


async def _build(activity):
    from app.services.prompting import defaults as d

    async def fake_get(key, **kwargs):
        assert key == "chat.offline_activity_section"
        return d.CHAT_OFFLINE_ACTIVITY_SECTION_PROMPT

    with patch(f"{PB}._get_optional_prompt", side_effect=fake_get):
        return await _build_offline_activity_section(activity)


@pytest.mark.asyncio
class TestOfflineActivitySection:
    async def test_none_activity_no_section(self):
        assert await _build(None) is None

    async def test_empty_title_no_section(self):
        assert await _build({"title": "", "location_name": "上河书房"}) is None

    async def test_reached_activity_injects_arrived_context(self):
        section = await _build(
            {"title": "古运河畔独处阅读", "location_name": "上河书房", "reached": True}
        )
        assert section is not None
        assert section.prompt_key == "chat.offline_activity_section"
        assert "古运河畔独处阅读" in section.body
        assert "上河书房" in section.body
        assert "已经到了现场" in section.body

    async def test_not_reached_activity_injects_enroute_context(self):
        section = await _build(
            {"title": "X 活动", "location_name": "Y 地点", "reached": False}
        )
        assert "前往" in section.body

    async def test_never_leaks_task_or_items(self):
        """硬约束: 段内含"严禁提及任务"的规则, 且模板本身不含任何拍摄物品名。"""
        section = await _build(
            {"title": "古运河畔独处阅读", "location_name": "上河书房", "reached": True}
        )
        body = section.body
        # 模板固定含 no-disclosure 硬约束
        assert "严禁" in body and "拍摄目标" in body
        # 绝不出现具体拍摄物品名（这些只应存在于 offline_shooting_conditions，不进聊天 prompt）
        for leak in ["运河岸柳", "阅读木椅", "街区招牌", "河畔步道"]:
            assert leak not in body

    async def test_disabled_template_removes_section(self):
        with patch(f"{PB}._get_optional_prompt", AsyncMock(return_value=None)):
            assert await _build_offline_activity_section(
                {"title": "X", "location_name": "Y", "reached": True}
            ) is None
