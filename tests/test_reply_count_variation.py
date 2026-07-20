"""图灵测试条数变化机制: 代码权威计数 + y 注入 + 系统标记全出口剥除.

三条保证 (用户需求):
1. 提示词 web 端定制版不动代码 default (新段 chat.reply_count_variation 是新 key)
2. 系统标记 ([EMO:]/[2]/【1】/[X]) 绝不暴露给用户
3. 多用户隔离 — Redis key 按 conversation_id 划分
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.chat.reply_count_state import (
    load_last_reply_count,
    save_last_reply_count,
)
from app.services.chat.reply_formatting import (
    split_and_validate_replies,
    strip_system_markers,
)

P = "app.services.chat.reply_count_state"


class _FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    async def set(self, key, value, ex=None):
        self.store[key] = str(value)

    async def get(self, key):
        return self.store.get(key)


# ── 1. Redis 状态: 存取 + 会话隔离 ──────────────────────────────────────


@pytest.mark.asyncio
class TestReplyCountState:
    async def test_save_and_load_roundtrip(self):
        fake = _FakeRedis()
        with patch(f"{P}.get_redis", AsyncMock(return_value=fake)):
            await save_last_reply_count("conv-1", 3)
            assert await load_last_reply_count("conv-1") == 3

    async def test_conversations_are_isolated(self):
        """多用户/多会话并发: 各自 key, 互不影响."""
        fake = _FakeRedis()
        with patch(f"{P}.get_redis", AsyncMock(return_value=fake)):
            await save_last_reply_count("conv-a", 1)
            await save_last_reply_count("conv-b", 4)
            assert await load_last_reply_count("conv-a") == 1
            assert await load_last_reply_count("conv-b") == 4
        assert set(fake.store) == {
            "reply:last_count:conv-a", "reply:last_count:conv-b",
        }

    async def test_no_record_returns_none(self):
        fake = _FakeRedis()
        with patch(f"{P}.get_redis", AsyncMock(return_value=fake)):
            assert await load_last_reply_count("conv-x") is None

    async def test_invalid_values_dropped(self):
        fake = _FakeRedis()
        fake.store["reply:last_count:conv-1"] = "garbage"
        fake.store["reply:last_count:conv-2"] = "99"  # 超出合理范围
        with patch(f"{P}.get_redis", AsyncMock(return_value=fake)):
            assert await load_last_reply_count("conv-1") is None
            assert await load_last_reply_count("conv-2") is None

    async def test_redis_down_degrades_silently(self):
        with patch(f"{P}.get_redis", AsyncMock(side_effect=RuntimeError("down"))):
            await save_last_reply_count("conv-1", 2)  # 不抛
            assert await load_last_reply_count("conv-1") is None


# ── 2. 系统标记剥除 (泄漏防护) ──────────────────────────────────────────


class TestStripSystemMarkers:
    def test_emo_marker_stripped_anywhere(self):
        assert strip_system_markers("好的呀 [EMO:高兴/70]") == "好的呀"
        assert strip_system_markers("[EMO:中性/50] 在呢") == "在呢"
        # 非法标签的畸形标记也剥 (宽松剥除原则)
        assert strip_system_markers("嗯 [EMO:生气/60]") == "嗯"

    def test_count_marker_whole_message_becomes_empty(self):
        """生产泄漏形态 1: '[2]' 单独成一条消息."""
        assert strip_system_markers("[2]") == ""
        assert strip_system_markers("【1】") == ""
        assert strip_system_markers("[X]") == ""

    def test_count_marker_trailing_stripped(self):
        """生产泄漏形态 2: 正文尾部带条数标记."""
        assert strip_system_markers("正在吃午饭呢 你呢？ [EMO:中性/50] 【1】") == "正在吃午饭呢 你呢？"
        assert strip_system_markers("好的 [3]") == "好的"

    def test_legit_brackets_preserved(self):
        """不误伤正常方括号内容: 表情文字 / 文中内嵌数字."""
        assert strip_system_markers("[捂脸] 太惨了") == "[捂脸] 太惨了"
        assert strip_system_markers("我住 [3] 号楼哈哈你猜") == "我住 [3] 号楼哈哈你猜"

    def test_split_pipeline_strips_markers(self):
        """split 管线 (_clean_reply_part) 集成剥除."""
        raw = "还没吃||你呢 [EMO:高兴/60]\n[2]"
        parts = split_and_validate_replies(raw)
        joined = " ".join(parts)
        assert "EMO" not in joined
        assert "[2]" not in joined


# ── 3. 条数上限 1-4 ────────────────────────────────────────────────────


class TestFourBubbleCap:
    def test_four_bubbles_pass_through(self):
        parts = split_and_validate_replies("一||二||三||四")
        assert parts == ["一", "二", "三", "四"]

    def test_fifth_bubble_dropped(self):
        parts = split_and_validate_replies("一||二||三||四||五")
        assert len(parts) == 4


# ── 4. prompt 段注入 ────────────────────────────────────────────────────


def _prompt_text(key: str) -> str:
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP

    definition = PROMPT_DEFINITION_MAP.get(key)
    return definition.default_text if definition else ""


def _patch_prompt_store():
    return (
        patch("app.services.chat.prompt_builder.get_prompt_text",
              AsyncMock(side_effect=_prompt_text)),
        patch("app.services.chat.prompt_builder.get_prompt_text_or_default",
              AsyncMock(side_effect=_prompt_text)),
    )


def _agent():
    return type("A", (), {"name": "小伴", "gender": "female",
                          "occupation": "客服", "city": "普洱"})()


@pytest.mark.asyncio
class TestVariationSection:
    async def test_section_rendered_with_y(self):
        from app.services.chat.prompt_builder import build_system_prompt

        p1, p2 = _patch_prompt_store()
        with p1, p2:
            prompt = await build_system_prompt(agent=_agent(), last_reply_count=3)
        assert "上一轮你回复了 3 条" in prompt
        assert "不能等于 3" in prompt

    async def test_section_skipped_without_history(self):
        from app.services.chat.prompt_builder import build_system_prompt

        p1, p2 = _patch_prompt_store()
        with p1, p2:
            prompt = await build_system_prompt(agent=_agent(), last_reply_count=None)
        assert "上一轮你回复了" not in prompt

    async def test_variation_prompt_not_in_reply_prefix(self):
        """新段绝不进 reply_prefix (前置需字节级稳定 + 其他路径无会话上下文)."""
        from app.services.prompting.reply_prefix import (
            PREFIX_SOURCE_KEYS,
            REPLY_PROMPT_KEYS,
        )

        assert "chat.reply_count_variation" not in PREFIX_SOURCE_KEYS
        assert "chat.reply_count_variation" not in REPLY_PROMPT_KEYS
