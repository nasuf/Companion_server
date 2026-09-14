"""featured_topics.py + topic_source.exclude_titles 参数 单测 (2026-09-14).

修 bug: admin 反复触发主动消息, DailyHot 热榜前几名短时间内不变 + V3 分类器
100% 确定性, 结果永远推"同一件事". 加"最近 featured 排除", 按 workspace_id
(即 user × agent) 隔离, 6h TTL. 不影响其他用户对同一热点的首次曝光.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.proactive.topic_source import classify_topic_source


class TestClassifierExcludeTitles:
    """topic_source.classify_topic_source 传 exclude_titles 时行为."""

    def _cand(self, title: str, snippet: str = "", platform: str = "微博") -> dict:
        return {"title": title, "snippet": snippet, "url": f"https://x.com/{title}",
                "platform": platform}

    def test_empty_exclude_backward_compatible(self):
        """未传 exclude_titles 或空集合 → 行为跟原来一模一样 (regression 保护)."""
        cands = [self._cand("热榜第一"), self._cand("热榜第二")]
        r1 = classify_topic_source(trending_candidates=cands)
        r2 = classify_topic_source(trending_candidates=cands, exclude_titles=frozenset())
        r3 = classify_topic_source(trending_candidates=cands, exclude_titles=set())
        # 三种调用方式结果一致
        assert r1.kind == r2.kind == r3.kind == "socially_hot"
        assert r1.selected_candidate == r2.selected_candidate == r3.selected_candidate

    def test_excluded_top_picks_next(self):
        """排除掉 socially_hot 会选的 top → 分类器自动选下一条."""
        cands = [
            self._cand("赵雷当爸爸"),   # 会被排除
            self._cand("iPhone 首销秒空"),
            self._cand("某明星综艺翻车"),
        ]
        r = classify_topic_source(
            trending_candidates=cands,
            exclude_titles={"赵雷当爸爸"},
        )
        assert r.kind == "socially_hot"
        assert r.selected_candidate is not None
        assert r.selected_candidate["title"] == "iPhone 首销秒空"

    def test_excluded_all_returns_none(self):
        """全部候选都在 exclude → kind='none' (兜底走 silence_plain)."""
        cands = [self._cand("A"), self._cand("B"), self._cand("C")]
        r = classify_topic_source(
            trending_candidates=cands,
            exclude_titles={"A", "B", "C"},
        )
        assert r.kind == "none"
        assert r.selected_candidate is None
        assert "全部" in r.reason and "featured" in r.reason.lower() or "featured" in r.reason

    def test_exclude_respects_user_interest_priority(self):
        """排除只对候选池, 不影响优先级. user_interest 命中的仍在 socially_hot 之上."""
        cands = [
            self._cand("A股大盘上涨"),     # 无匹配 → socially_hot
            self._cand("摄影展开幕", "国家美术馆摄影展周末开幕"),  # 命中用户"摄影"
        ]
        r = classify_topic_source(
            trending_candidates=cands,
            user_portrait="用户喜欢摄影和爬山",
            exclude_titles={"A股大盘上涨"},  # 排除 hot 候选
        )
        # user_interest 那条还在, 应仍走 user_interest_match
        assert r.kind == "user_interest_match"
        assert r.selected_candidate["title"] == "摄影展开幕"

    def test_exclude_diagnostic_in_reason(self):
        """排除动作在 reason 里留痕, 方便 log 排查."""
        cands = [self._cand("旧闻"), self._cand("新闻")]
        r = classify_topic_source(
            trending_candidates=cands,
            exclude_titles={"旧闻"},
        )
        assert "排除 1 条" in r.reason and "featured" in r.reason


class TestFeaturedTopicsRedis:
    """featured_topics.py Redis CRUD."""

    @pytest.mark.asyncio
    async def test_get_empty_when_no_workspace(self):
        from app.services.proactive.featured_topics import get_recent_featured
        assert await get_recent_featured("") == set()
        assert await get_recent_featured(None) == set()

    @pytest.mark.asyncio
    async def test_get_reads_and_parses(self):
        from app.services.proactive import featured_topics as m
        redis = MagicMock()
        redis.lrange = AsyncMock(return_value=[
            "赵雷当爸爸".encode(),
            "iPhone 秒空".encode(),
            "第三条",
        ])
        with patch.object(m, "get_redis", new=AsyncMock(return_value=redis)):
            result = await m.get_recent_featured("ws-1")
        assert result == {"赵雷当爸爸", "iPhone 秒空", "第三条"}

    @pytest.mark.asyncio
    async def test_get_swallows_redis_failure(self):
        """Redis 挂 → 返空集合 (放行, 不阻塞主动消息发送)."""
        from app.services.proactive import featured_topics as m
        redis = MagicMock()
        redis.lrange = AsyncMock(side_effect=RuntimeError("redis down"))
        with patch.object(m, "get_redis", new=AsyncMock(return_value=redis)):
            assert await m.get_recent_featured("ws-1") == set()

    @pytest.mark.asyncio
    async def test_remember_writes_and_trims(self):
        from app.services.proactive import featured_topics as m
        pipe = MagicMock()
        pipe.lpush = MagicMock(); pipe.ltrim = MagicMock(); pipe.expire = MagicMock()
        pipe.execute = AsyncMock()
        redis = MagicMock()
        redis.pipeline = MagicMock(return_value=pipe)
        with patch.object(m, "get_redis", new=AsyncMock(return_value=redis)):
            await m.remember_featured("ws-1", "赵雷当爸爸")
        pipe.lpush.assert_called_once()
        pipe.ltrim.assert_called_once_with(
            m._FEATURED_KEY.format(workspace_id="ws-1"),
            0, m._FEATURED_MAX - 1,
        )
        pipe.expire.assert_called_once_with(
            m._FEATURED_KEY.format(workspace_id="ws-1"),
            m._FEATURED_TTL_S,
        )
        pipe.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_remember_no_op_on_empty(self):
        from app.services.proactive import featured_topics as m
        redis = MagicMock()
        redis.pipeline = MagicMock()  # 不该被调
        with patch.object(m, "get_redis", new=AsyncMock(return_value=redis)):
            await m.remember_featured("", "some title")
            await m.remember_featured("ws-1", "")
        redis.pipeline.assert_not_called()

    @pytest.mark.asyncio
    async def test_per_user_isolation(self):
        """workspace_A 记账不影响 workspace_B 的读取 (per-user 隔离验证)."""
        from app.services.proactive import featured_topics as m
        # 模拟两个 workspace 的 Redis 键各自独立
        store: dict[str, list] = {}

        async def fake_lrange(key, start, end):
            return [t.encode() for t in store.get(key, [])[start:end + 1]]

        async def fake_pipe_exec():
            return [None, None, None]

        pipe_a = MagicMock()
        pipe_a.execute = AsyncMock(side_effect=fake_pipe_exec)
        def _lpush_a(key, val):
            store.setdefault(key, []).insert(0, val)
        pipe_a.lpush = MagicMock(side_effect=_lpush_a)
        pipe_a.ltrim = MagicMock()
        pipe_a.expire = MagicMock()

        redis = MagicMock()
        redis.lrange = AsyncMock(side_effect=fake_lrange)
        redis.pipeline = MagicMock(return_value=pipe_a)

        with patch.object(m, "get_redis", new=AsyncMock(return_value=redis)):
            # user A 记了一个热点
            await m.remember_featured("ws-user-A", "赵雷当爸爸")
            # user B 读自己的 → 空 (关键验证: 隔离)
            b_result = await m.get_recent_featured("ws-user-B")
            # user A 读自己的 → 有
            a_result = await m.get_recent_featured("ws-user-A")

        assert b_result == set(), "user B 不该看到 user A 的 featured 记账"
        assert a_result == {"赵雷当爸爸"}
