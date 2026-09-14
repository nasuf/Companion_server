"""proactive/recent_messages.py 单测.

覆盖 anti-repetition 守卫: 完全一致 / 高度相似 / 无关 / Redis 挂 / 空输入.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestShinglingSimilarity:
    def test_shingles_two_char_window(self):
        from app.services.proactive.recent_messages import _shingles
        assert _shingles("最近咋样") == {"最近", "近咋", "咋样"}
        assert _shingles("在") == {"在"}  # 单字 → 就它自己
        assert _shingles("") == set()

    def test_similarity_exact_match(self):
        from app.services.proactive.recent_messages import _similarity
        assert _similarity("最近好像大家都在聊赵雷当爸爸了",
                           "最近好像大家都在聊赵雷当爸爸了") == 1.0

    def test_similarity_high_overlap(self):
        from app.services.proactive.recent_messages import _similarity
        # 主体相同, 换一两个字 → 相似度仍高
        sim = _similarity(
            "最近好像大家都在聊赵雷当爸爸了",
            "最近好像大家都在聊赵雷当奶爸",
        )
        assert sim > 0.5

    def test_similarity_different_content_low(self):
        from app.services.proactive.recent_messages import _similarity
        sim = _similarity(
            "最近好像大家都在聊赵雷当爸爸了",
            "最近好像大家都在聊 iPhone 17 首销秒空",
        )
        assert sim < 0.5

    def test_similarity_empty_returns_zero(self):
        from app.services.proactive.recent_messages import _similarity
        assert _similarity("", "abc") == 0.0
        assert _similarity("abc", "") == 0.0


class TestIsRepeatOfRecent:
    @pytest.mark.asyncio
    async def test_returns_false_when_workspace_empty(self):
        from app.services.proactive.recent_messages import is_repeat_of_recent
        assert not await is_repeat_of_recent("", "msg")
        assert not await is_repeat_of_recent(None, "msg")

    @pytest.mark.asyncio
    async def test_returns_false_when_text_empty(self):
        from app.services.proactive.recent_messages import is_repeat_of_recent
        assert not await is_repeat_of_recent("ws-1", "")

    @pytest.mark.asyncio
    async def test_returns_true_on_exact_recent_match(self):
        from app.services.proactive import recent_messages as m
        redis = MagicMock()
        redis.lrange = AsyncMock(return_value=[
            json.dumps({"text": "最近好像大家都在聊赵雷当爸爸了"}).encode(),
        ])
        with patch.object(m, "get_redis", new=AsyncMock(return_value=redis)):
            assert await m.is_repeat_of_recent(
                "ws-1", "最近好像大家都在聊赵雷当爸爸了",
            )

    @pytest.mark.asyncio
    async def test_returns_false_when_no_recent_similar(self):
        from app.services.proactive import recent_messages as m
        redis = MagicMock()
        redis.lrange = AsyncMock(return_value=[
            json.dumps({"text": "早呀，今天天气不错"}).encode(),
        ])
        with patch.object(m, "get_redis", new=AsyncMock(return_value=redis)):
            assert not await m.is_repeat_of_recent(
                "ws-1", "最近好像大家都在聊赵雷当爸爸了",
            )

    @pytest.mark.asyncio
    async def test_returns_false_on_redis_failure(self):
        # Redis 挂了不能阻塞发送 —— 宁可放行也不 raise
        from app.services.proactive import recent_messages as m
        redis = MagicMock()
        redis.lrange = AsyncMock(side_effect=RuntimeError("redis down"))
        with patch.object(m, "get_redis", new=AsyncMock(return_value=redis)):
            assert not await m.is_repeat_of_recent("ws-1", "任意消息")

    @pytest.mark.asyncio
    async def test_returns_false_on_empty_redis_list(self):
        from app.services.proactive import recent_messages as m
        redis = MagicMock()
        redis.lrange = AsyncMock(return_value=[])
        with patch.object(m, "get_redis", new=AsyncMock(return_value=redis)):
            assert not await m.is_repeat_of_recent("ws-1", "任意消息")

    @pytest.mark.asyncio
    async def test_tolerates_malformed_recent_entry(self):
        # Redis 里坏 JSON 应跳过, 不阻塞其它比较
        from app.services.proactive import recent_messages as m
        redis = MagicMock()
        redis.lrange = AsyncMock(return_value=[
            b"not json",
            json.dumps({"text": "最近好像大家都在聊赵雷"}).encode(),
        ])
        with patch.object(m, "get_redis", new=AsyncMock(return_value=redis)):
            # 第二条应能命中
            assert await m.is_repeat_of_recent(
                "ws-1", "最近好像大家都在聊赵雷",
            )


class TestRememberRecent:
    @pytest.mark.asyncio
    async def test_no_op_on_empty_input(self):
        from app.services.proactive import recent_messages as m
        # 空 workspace / 空文本都不该发起 redis 调用
        redis = MagicMock()
        redis.pipeline = MagicMock()  # 不该被调
        with patch.object(m, "get_redis", new=AsyncMock(return_value=redis)):
            await m.remember_recent("", "some msg")
            await m.remember_recent("ws-1", "")
        redis.pipeline.assert_not_called()

    @pytest.mark.asyncio
    async def test_writes_and_trims(self):
        from app.services.proactive import recent_messages as m
        pipe = MagicMock()
        pipe.lpush = MagicMock(); pipe.ltrim = MagicMock(); pipe.expire = MagicMock()
        pipe.execute = AsyncMock()
        redis = MagicMock()
        redis.pipeline = MagicMock(return_value=pipe)
        with patch.object(m, "get_redis", new=AsyncMock(return_value=redis)):
            await m.remember_recent("ws-1", "新消息")
        pipe.lpush.assert_called_once()
        pipe.ltrim.assert_called_once_with(m._RECENT_KEY.format(workspace_id="ws-1"),
                                            0, m._RECENT_MAX - 1)
        pipe.expire.assert_called_once()
        pipe.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_swallows_redis_failure(self):
        # Redis 写失败不能拖崩主流程
        from app.services.proactive import recent_messages as m
        redis = MagicMock()
        redis.pipeline = MagicMock(side_effect=RuntimeError("redis down"))
        with patch.object(m, "get_redis", new=AsyncMock(return_value=redis)):
            await m.remember_recent("ws-1", "msg")  # 不 raise 即可
