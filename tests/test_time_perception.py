"""Phase B 时间感知回归测试.

B1: 历史消息注入绝对时间前缀 [MM-DD HH:MM] (UTC+8, cache 友好)
B2: 重逢感知段 — ≥30min 间隔按三档注入分级叙事
B3: ≥3h 间隔清空话题栈, 防旧话题拖过长间隔
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.chat.prompt_builder import (
    _build_reengagement_section,
    build_chat_messages,
    compute_reengagement_gap_seconds,
    format_gap_text,
    format_message_timestamp,
)

PB = "app.services.chat.prompt_builder"


# ─────────────────────────────────────────────────────────────────
# B1: format_message_timestamp / build_chat_messages
# ─────────────────────────────────────────────────────────────────


class TestFormatMessageTimestamp:
    def test_utc_iso_converts_to_utc8(self):
        # UTC 03:05 = 北京时间 11:05
        assert format_message_timestamp("2026-07-03T03:05:00+00:00") == "[07-03 11:05] "

    def test_naive_datetime_treated_as_utc(self):
        assert format_message_timestamp("2026-07-03T03:05:00") == "[07-03 11:05] "

    def test_datetime_object_accepted(self):
        dt = datetime(2026, 7, 3, 3, 5, tzinfo=timezone.utc)
        assert format_message_timestamp(dt) == "[07-03 11:05] "

    def test_none_and_invalid_return_empty(self):
        assert format_message_timestamp(None) == ""
        assert format_message_timestamp("") == ""
        assert format_message_timestamp("not-a-date") == ""

    def test_deterministic_absolute_not_relative(self):
        """cache 稳定性契约: 同一 createdAt 任何时刻渲染结果字节级一致.

        (MaiBot 的"5分钟前"式相对时间每轮都变, 会把历史段 prompt cache 打穿 —
        我们刻意用绝对时间, 此测试防止未来改回相对时间.)
        """
        ts = "2026-07-01T10:00:00+00:00"
        assert format_message_timestamp(ts) == format_message_timestamp(ts)
        assert "前" not in format_message_timestamp(ts)


class TestBuildChatMessagesTimestamps:
    def test_history_gets_timestamp_prefix(self):
        msgs = [
            {"role": "user", "content": "早上好", "createdAt": "2026-07-03T00:00:00+00:00"},
            {"role": "assistant", "content": "早呀", "createdAt": "2026-07-03T00:00:30+00:00"},
        ]
        out = build_chat_messages("SYS", msgs)
        assert out[0] == {"role": "system", "content": "SYS"}
        assert out[1]["content"] == "[07-03 08:00] 早上好"
        assert out[2]["content"] == "[07-03 08:00] 早呀"

    def test_message_without_created_at_has_no_prefix(self):
        out = build_chat_messages("SYS", [{"role": "user", "content": "hi"}])
        assert out[1]["content"] == "hi"

    def test_two_builds_are_identical(self):
        """cache 稳定性: 同一输入两次构建输出完全一致."""
        msgs = [
            {"role": "user", "content": "a", "createdAt": "2026-07-01T10:00:00+00:00"},
        ]
        assert build_chat_messages("SYS", msgs) == build_chat_messages("SYS", msgs)


# ─────────────────────────────────────────────────────────────────
# B2: gap 计算 + 分档注入
# ─────────────────────────────────────────────────────────────────


class TestFormatGapText:
    def test_minutes(self):
        assert format_gap_text(45 * 60) == "45 分钟"
        assert format_gap_text(30) == "1 分钟"  # 下限保护

    def test_hours(self):
        assert format_gap_text(5 * 3600) == "5 小时"
        assert format_gap_text(23 * 3600) == "23 小时"  # <24h 用小时
        assert format_gap_text(30 * 3600) == "1 天"  # ≥24h 用天 (review 修复: 与"隔天"档叙事一致)

    def test_days(self):
        assert format_gap_text(3 * 86400) == "3 天"


class TestComputeReengagementGap:
    NOW = datetime(2026, 7, 3, 12, 0, tzinfo=timezone.utc)

    def _msg(self, mid, minutes_ago, role="user"):
        return {
            "id": mid,
            "role": role,
            "content": "x",
            "createdAt": (self.NOW - timedelta(minutes=minutes_ago)).isoformat(),
        }

    def test_gap_from_last_previous_turn_message(self):
        msgs = [self._msg("m1", 300), self._msg("cur", 0)]
        gap = compute_reengagement_gap_seconds(msgs, exclude_ids={"cur"}, now=self.NOW)
        assert gap == pytest.approx(300 * 60)

    def test_excludes_all_current_turn_ids(self):
        """聚合轮次: 当前轮多条消息全部排除, gap 取自上一轮."""
        msgs = [self._msg("old", 240), self._msg("c1", 0.02), self._msg("c2", 0.01)]
        gap = compute_reengagement_gap_seconds(
            msgs, exclude_ids={"c1", "c2"}, now=self.NOW,
        )
        assert gap == pytest.approx(240 * 60)

    def test_grace_window_skips_fresh_synthetic_message(self):
        """合成当前消息无 id 进不了 exclude — 距 now <10s 兜底跳过."""
        synthetic = {
            "id": None, "role": "user", "content": "x",
            "createdAt": (self.NOW - timedelta(seconds=2)).isoformat(),
        }
        msgs = [self._msg("old", 120), synthetic]
        gap = compute_reengagement_gap_seconds(msgs, exclude_ids=set(), now=self.NOW)
        assert gap == pytest.approx(120 * 60)

    def test_no_history_returns_none(self):
        assert compute_reengagement_gap_seconds([], now=self.NOW) is None
        only_current = [self._msg("cur", 0)]
        assert compute_reengagement_gap_seconds(
            only_current, exclude_ids={"cur"}, now=self.NOW,
        ) is None


@pytest.mark.asyncio
class TestReengagementSection:
    """分档: <30min 无 / 30min-3h short / 3h-24h long / >24h day."""

    async def _build(self, gap_seconds):
        # _get_optional_prompt 直连 registry 默认模板 (store 不可用时的默认行为)
        from app.services.prompting import defaults as d

        tpl_by_key = {
            "chat.reengagement_short": d.CHAT_REENGAGEMENT_SHORT_PROMPT,
            "chat.reengagement_long": d.CHAT_REENGAGEMENT_LONG_PROMPT,
            "chat.reengagement_day": d.CHAT_REENGAGEMENT_DAY_PROMPT,
        }

        async def fake_get(key, **kwargs):
            return tpl_by_key[key]

        with patch(f"{PB}._get_optional_prompt", side_effect=fake_get):
            return await _build_reengagement_section(gap_seconds)

    async def test_below_30min_no_section(self):
        assert await self._build(None) is None
        assert await self._build(29 * 60) is None

    async def test_short_tier(self):
        section = await self._build(60 * 60)
        assert section.prompt_key == "chat.reengagement_short"
        assert "1 小时" in section.body

    async def test_long_tier(self):
        section = await self._build(5 * 3600)
        assert section.prompt_key == "chat.reengagement_long"
        assert "5 小时" in section.body
        assert "重新开始" in section.body

    async def test_day_tier(self):
        section = await self._build(3 * 86400)
        assert section.prompt_key == "chat.reengagement_day"
        assert "3 天" in section.body

    async def test_disabled_template_removes_section(self):
        """admin 停用模板 → 该档整段不注入 (registry 停用语义)."""
        with patch(f"{PB}._get_optional_prompt", AsyncMock(return_value=None)):
            assert await _build_reengagement_section(5 * 3600) is None


# ─────────────────────────────────────────────────────────────────
# B3: 长间隔清空话题栈
# ─────────────────────────────────────────────────────────────────


class _FakeTopicRedis:
    def __init__(self, top_entry: dict | None = None):
        self.stack: list[str] = (
            [json.dumps(top_entry, ensure_ascii=False)] if top_entry else []
        )
        self.deleted = False

    async def delete(self, key):
        self.stack.clear()
        self.deleted = True

    async def lindex(self, key, i):
        return self.stack[i] if self.stack else None

    async def lset(self, key, i, value):
        self.stack[i] = value

    async def lpush(self, key, value):
        self.stack.insert(0, value)

    async def ltrim(self, key, a, b):
        del self.stack[b + 1:]

    async def expire(self, key, ttl):
        pass


@pytest.mark.asyncio
class TestTopicResetOnReengagement:
    async def test_long_gap_resets_topic_stack(self):
        from app.services.topic import TOPIC_RESET_GAP_SECONDS, push_topic

        fake = _FakeTopicRedis(top_entry={"topic": "旧话题", "turns": 7, "category": "工作"})
        with patch("app.services.topic.get_redis", AsyncMock(return_value=fake)):
            entry = await push_topic("c1", "今天上班好累", gap_seconds=TOPIC_RESET_GAP_SECONDS + 1)
        assert fake.deleted  # 旧栈被清
        assert entry["turns"] == 1  # 当前消息是全新话题, 不继承 7 轮计数

    async def test_short_gap_keeps_topic_continuity(self):
        from app.services.topic import push_topic

        fake = _FakeTopicRedis(top_entry={"topic": "旧话题", "turns": 7, "category": "工作"})
        with patch("app.services.topic.get_redis", AsyncMock(return_value=fake)):
            entry = await push_topic("c1", "老板今天又开会", gap_seconds=60.0)
        assert not fake.deleted
        assert entry["turns"] == 8  # 同类话题正常累计

    async def test_none_gap_keeps_old_behavior(self):
        from app.services.topic import push_topic

        fake = _FakeTopicRedis(top_entry={"topic": "旧话题", "turns": 2, "category": "工作"})
        with patch("app.services.topic.get_redis", AsyncMock(return_value=fake)):
            entry = await push_topic("c1", "加班真多", gap_seconds=None)
        assert not fake.deleted
        assert entry["turns"] == 3
