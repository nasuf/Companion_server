"""W3 回归: 关系时长感知 MVP."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.relationship.relation_meta import (
    format_relation_meta_line,
    get_relation_meta,
)

P = "app.services.relationship.relation_meta"


class TestFormatLine:
    def test_normal_days_and_rounded_turns(self):
        line = format_relation_meta_line({"days_known": 42, "approx_turns": 327})
        assert "认识 42 天" in line
        assert "大约 320 轮" in line  # 向下取整到 10

    def test_first_day_special_wording(self):
        line = format_relation_meta_line({"days_known": 0, "approx_turns": 5})
        assert "今天刚认识" in line
        assert "5 轮" in line

    def test_none_meta_renders_empty(self):
        assert format_relation_meta_line(None) == ""


@pytest.mark.asyncio
class TestGetRelationMeta:
    async def test_cache_hit_skips_db(self):
        created = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        fake_redis = AsyncMock()
        fake_redis.get = AsyncMock(return_value=json.dumps(
            {"created_at": created, "approx_turns": 100},
        ))
        fake_db = MagicMock()
        with (
            patch(f"{P}.get_redis", AsyncMock(return_value=fake_redis)),
            patch(f"{P}.db", fake_db),
        ):
            meta = await get_relation_meta("c1")
        assert meta == {"days_known": 10, "approx_turns": 100}
        fake_db.conversation.find_unique.assert_not_called()

    async def test_cache_miss_queries_db_and_caches(self):
        fake_redis = AsyncMock()
        fake_redis.get = AsyncMock(return_value=None)
        conv = MagicMock()
        conv.createdAt = datetime.now(timezone.utc) - timedelta(days=3)
        fake_db = MagicMock()
        fake_db.conversation.find_unique = AsyncMock(return_value=conv)
        fake_db.message.count = AsyncMock(return_value=90)
        with (
            patch(f"{P}.get_redis", AsyncMock(return_value=fake_redis)),
            patch(f"{P}.db", fake_db),
        ):
            meta = await get_relation_meta("c1")
        assert meta == {"days_known": 3, "approx_turns": 30}  # 90 条 ≈ 30 轮
        fake_redis.set.assert_awaited_once()

    async def test_db_failure_returns_none(self):
        fake_redis = AsyncMock()
        fake_redis.get = AsyncMock(return_value=None)
        fake_db = MagicMock()
        fake_db.conversation.find_unique = AsyncMock(side_effect=RuntimeError("db down"))
        with (
            patch(f"{P}.get_redis", AsyncMock(return_value=fake_redis)),
            patch(f"{P}.db", fake_db),
        ):
            assert await get_relation_meta("c1") is None

    async def test_no_conversation_id_returns_none(self):
        assert await get_relation_meta(None) is None


@pytest.mark.asyncio
async def test_relationship_section_includes_meta_line():
    from app.services.chat.prompt_builder import _build_emotion_section
    from app.services.prompting import defaults as d

    with patch(
        "app.services.chat.prompt_builder._get_optional_prompt",
        AsyncMock(return_value=d.CHAT_RELATIONSHIP_STAGE_SECTION_PROMPT),
    ):
        section = await _build_emotion_section(
            None, "熟悉期",
            relation_meta_line="你们认识 42 天了，聊过大约 320 轮。",
        )
        assert "认识 42 天" in section.body
        assert "不用主动复述数字" in section.body

        # 无 meta 时占位符原地消失, 段仍正常
        section_bare = await _build_emotion_section(None, "熟悉期")
        assert "认识" not in section_bare.body
        assert "熟悉期" in section_bare.body
