"""惰性效用值更新的行为守卫.

这套机制取代了夜间全表重算, 所以它必须比被取代者更难悄悄坏掉 —— 旧 cron 死了
几个月无人察觉, 正是因为没有任何测试盯着"分数到底有没有在动"。
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from app.services.memory.lifecycle.value import (
    ACCESS_CEILING,
    ACCESS_REWARD,
    CONTRIBUTION_REWARD,
    DECAY_LAMBDA,
    HALF_LIFE_DAYS,
    HOT_DEMOTE_AT,
    HOT_PROMOTE_AT,
    VALUE_MAX,
    WARM_DEMOTE_AT,
    WARM_PROMOTE_AT,
    apply_usage,
    days_since,
    decayed_value,
    next_level,
)


def test_persona_fade_time_matches_what_life_story_documents():
    """人设分层那段注释写了具体的淡出天数, 它依赖半衰期 —— 两边不能各说各的。

    历史教训: 那段注释原先按旧的分段档位公式写"约一年", Phase 1 换成指数衰减后
    实际只有 140 天, 而注释一直没改。判断"人设会不会消失得太快"的人会照着注释
    做决定。
    """
    import math

    from app.services.memory.lifecycle.value import DECAY_LAMBDA, WARM_DEMOTE_AT

    days = math.log(0.72 / WARM_DEMOTE_AT) / DECAY_LAMBDA
    assert 150 <= days <= 260, (
        f"最低档人设 (0.72) 现在 {days:.0f} 天跌到 L3；"
        "life_story.py 里写的区间要跟着改"
    )


class TestDecay:
    def test_half_life_is_what_the_constant_says(self):
        assert decayed_value(1.0, HALF_LIFE_DAYS) == pytest.approx(0.5, abs=1e-9)

    def test_no_elapsed_time_means_no_decay(self):
        """幂等的基础: 同一时刻重复更新不该反复扣分。"""
        assert decayed_value(0.7, 0) == 0.7
        assert decayed_value(0.7, -5) == 0.7

    def test_decay_is_monotonic(self):
        previous = 1.0
        for days in (1, 10, 100, 1000):
            current = decayed_value(1.0, days)
            assert current < previous
            previous = current


class TestRewards:
    def test_contribution_outweighs_access(self):
        """AMV-L 要求 β ≥ α: 真正进了 prompt 比"进过候选"是更强的效用证据。"""
        assert CONTRIBUTION_REWARD > ACCESS_REWARD

    def test_reward_applies_after_decay_not_before(self):
        """顺序反了会把刚拿到的回报也打折, 让高频使用的记忆分数系统性偏低。

        用半衰期本身作为经过时间, 这样断言不依赖具体常数 —— 衰减恰好折半, 回报
        原样加上去。
        """
        result = apply_usage(
            value=0.5, level=2, days_idle=HALF_LIFE_DAYS, contributed=True,
        )
        assert result.value == pytest.approx(0.25 + CONTRIBUTION_REWARD, abs=1e-6)

    def test_access_alone_can_never_reach_the_hot_band(self):
        """仅仅"被向量检索捞到过"不该让一条记忆变成核心记忆。

        标定时发现纯加法回报做不到这点: α=0.05 配 180 天半衰期时, 每 30 天进一次
        候选集就能一路涨到上限。改成趋向天花板的递减回报后才有这个性质。
        """
        value, level = 0.1, 3
        for _ in range(500):
            result = apply_usage(
                value=value, level=level, days_idle=0.5, accessed=True,
            )
            value, level = result.value, result.level
        assert value <= ACCESS_CEILING + 1e-9
        assert value < HOT_PROMOTE_AT
        assert level == 2, "只进候选集却升到了 hot"

    def test_contribution_can_reach_the_hot_band(self):
        """真正被注入 prompt 必须能推着记忆穿过 hot 阈值, 否则升级路径又形同虚设。"""
        value, level = 0.1, 3
        for _ in range(50):
            result = apply_usage(
                value=value, level=level, days_idle=0.5, contributed=True,
            )
            value, level = result.value, result.level
        assert level == 1

    def test_access_does_not_drag_down_an_already_hot_memory(self):
        """递减回报在分数高于天花板时应为 0, 不能变成惩罚。"""
        high = apply_usage(value=0.95, level=1, days_idle=0, accessed=True)
        assert high.value == pytest.approx(0.95)

    def test_value_is_capped(self):
        result = apply_usage(value=0.99, level=1, days_idle=0, contributed=True)
        assert result.value <= VALUE_MAX

    def test_value_never_goes_negative(self):
        result = apply_usage(value=0.0, level=3, days_idle=10_000)
        assert result.value >= 0.0


class TestHysteresis:
    def test_promote_threshold_sits_above_demote_threshold(self):
        """没有死区就会在阈值附近反复横跳, 每跳一次都要写库。"""
        assert HOT_PROMOTE_AT > HOT_DEMOTE_AT
        assert WARM_PROMOTE_AT > WARM_DEMOTE_AT

    def test_dead_zone_is_crossable_by_one_real_use(self):
        """死区太宽会让记忆升不上去。一次 contribution 应当足以穿过。"""
        assert (HOT_PROMOTE_AT - HOT_DEMOTE_AT) < CONTRIBUTION_REWARD
        assert (WARM_PROMOTE_AT - WARM_DEMOTE_AT) < CONTRIBUTION_REWARD

    def test_value_inside_dead_zone_keeps_current_level(self):
        middle = (HOT_PROMOTE_AT + HOT_DEMOTE_AT) / 2
        assert next_level(middle, 1) == 1
        assert next_level(middle, 2) == 2

    def test_same_value_keeps_different_levels_depending_on_history(self):
        """这就是滞回的定义: 层级取决于从哪一侧进入死区, 而不只是当前分数。

        没有这个性质, 一条分数在阈值附近游走的记忆会被反复升降, 每跳一次都要写库,
        用户也会觉得 AI 时而记得时而不记得。
        """
        inside = (HOT_PROMOTE_AT + HOT_DEMOTE_AT) / 2
        assert next_level(inside, 1) == 1
        assert next_level(inside, 2) == 2

    def test_wiggling_inside_the_dead_zone_never_flips_the_level(self):
        """分数在死区内小幅上下, 层级必须岿然不动。"""
        span = HOT_PROMOTE_AT - HOT_DEMOTE_AT
        for start_level in (1, 2):
            levels = {
                next_level(HOT_DEMOTE_AT + span * frac, start_level)
                for frac in (0.05, 0.3, 0.5, 0.7, 0.95)
            }
            assert levels == {start_level}, f"L{start_level} 在死区内抖动: {levels}"


class TestLevelTransitions:
    def test_cold_memory_can_return_to_warm(self):
        """旧实现没有 L3→L2 的路径, 掉下去就永远回不来。"""
        assert next_level(WARM_PROMOTE_AT, 3) == 2

    def test_identity_facts_never_leave_l1(self):
        """用户一年没问过"你叫什么", 不代表 AI 可以不知道自己叫什么。"""
        assert next_level(0.0, 1, protected=True) == 1

    def test_unprotected_hot_memory_can_cool_off(self):
        assert next_level(HOT_DEMOTE_AT - 0.01, 1) == 2

    def test_promotion_does_not_require_user_emphasis(self):
        """旧实现要求"用户曾标记重要", 历史上 0 次升级 —— 那等于没有升级路径。
        新规则是纯值驱动的。"""
        result = apply_usage(value=HOT_PROMOTE_AT, level=2, days_idle=0)
        assert result.level == 1
        assert result.changed_level


class TestDaysSince:
    def test_missing_timestamps_mean_no_decay(self):
        """时间基准缺失时宁可少衰减, 也不要凭空把记忆打入冷宫。"""
        assert days_since(None, None) == 0.0

    def test_falls_back_to_the_secondary_anchor(self):
        created = datetime.now(UTC) - timedelta(days=10)
        assert days_since(None, created) == pytest.approx(10, abs=0.01)

    def test_naive_datetimes_are_treated_as_utc(self):
        naive = (datetime.now(UTC) - timedelta(days=5)).replace(tzinfo=None)
        assert days_since(naive) == pytest.approx(5, abs=0.01)


class TestRecordMemoryUsage:
    @pytest.mark.asyncio
    async def test_contribution_wins_over_access_for_the_same_memory(self):
        """注入本来就蕴含"进过候选", 叠加等于给同一件事记两次功。"""
        from app.services.memory.lifecycle.lazy_update import _signals

        signals = _signals(["m1"], ["m1", "m2"])
        assert signals["m1"] is True, "m1 被注入过, 应按 contribution 计"
        assert signals["m2"] is False

    @pytest.mark.asyncio
    async def test_empty_input_touches_no_database(self):
        from app.services.memory.lifecycle import lazy_update

        with patch.object(lazy_update.db, "execute_raw", new=AsyncMock()) as raw:
            assert await lazy_update.record_memory_usage() == 0
            raw.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_database_failure_never_propagates(self):
        from app.services.memory.lifecycle import lazy_update
        with patch.object(lazy_update, "_apply_batch", new=AsyncMock(side_effect=RuntimeError("db down"))):
            assert await lazy_update.record_memory_usage(event_id="event", user_id="user", contributed_ids=["m1"]) == 0

    @pytest.mark.asyncio
    async def test_both_sides_receive_stable_event_and_scope(self):
        from app.services.memory.lifecycle import lazy_update
        with patch.object(lazy_update, "_apply_batch", new=AsyncMock(return_value={"total": 1})) as batch:
            assert await lazy_update.record_memory_usage(event_id="event", user_id="user", workspace_id="scope", contributed_ids=["m1"]) == 2
        assert {c.kwargs["source"] for c in batch.await_args_list} == {"user", "ai"}
        assert all(c.kwargs["event_id"] == "event" and c.kwargs["workspace_id"] == "scope" for c in batch.await_args_list)

    @pytest.mark.asyncio
    async def test_missing_event_cannot_reward(self):
        from app.services.memory.lifecycle import lazy_update
        with patch.object(lazy_update, "_apply_batch", new=AsyncMock()) as batch:
            assert await lazy_update.record_memory_usage(user_id="user", contributed_ids=["m1"]) == 0
        batch.assert_not_awaited()


class TestSweepFailureVisibility:
    @pytest.mark.asyncio
    async def test_failure_propagates_to_scheduler(self):
        from app.services.memory.lifecycle import lazy_update
        with patch.object(lazy_update.db, "query_raw", new=AsyncMock(side_effect=RuntimeError("boom"))):
            with pytest.raises(RuntimeError, match="boom"):
                await lazy_update.sweep_stale_values()

    @pytest.mark.asyncio
    async def test_invalid_batch_is_rejected(self):
        from app.services.memory.lifecycle import lazy_update
        with pytest.raises(ValueError):
            await lazy_update.sweep_stale_values(limit=0)


def test_half_life_change_requires_rerunning_the_simulation():
    """常数漂了而没重跑推演, 等于闸门结论作废。这里钉住当前值。

    240 是推演选出来的: 180 天在存量人设重排后的最坏场景第 180 天有 -4.2% 退化
    (人设落在 0.72-0.82, 约 140 天就跌破 warm 下行阈值), 240 是消除它的最小值。
    """
    assert HALF_LIFE_DAYS == 240.0
    assert DECAY_LAMBDA == pytest.approx(math.log(2) / 240.0)
