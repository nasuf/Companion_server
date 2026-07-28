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


class TestSqlMatchesPython:
    """SQL 里的算术必须和 Python 纯函数算出同一个数。

    两处实现分别服务热路径 (SQL, 免读-改-写竞态) 和离线推演 (Python), 一旦漂移,
    推演的结论就不适用于生产 —— 而推演正是我们判断"改了会不会变差"的唯一依据。
    这里用符号比对常量, 真值比对靠下面的 SQL 结构断言。
    """

    def test_sql_embeds_the_same_constants(self):
        from app.services.memory.lifecycle import lazy_update

        sql = lazy_update._render_sql("memories_user")
        for constant in (DECAY_LAMBDA, HOT_PROMOTE_AT, WARM_DEMOTE_AT,
                         ACCESS_CEILING, CONTRIBUTION_REWARD):
            assert str(constant) in sql, f"{constant} 没进 SQL"

    def test_sql_distinguishes_the_two_signals(self):
        """两种信号形式不同 —— SQL 里必须体现, 否则 access 又能把记忆推进 hot。"""
        from app.services.memory.lifecycle import lazy_update

        sql = lazy_update._render_sql("memories_user")
        assert "u.is_contribution" in sql
        assert f"{ACCESS_REWARD} * GREATEST(0.0, {ACCESS_CEILING}" in sql

    def test_sql_decays_before_rewarding(self):
        """`decayed + reward` 而不是 `(base + reward) * EXP(...)`。"""
        from app.services.memory.lifecycle import lazy_update

        sql = lazy_update._render_sql("memories_user")
        assert f"d.decayed + {CONTRIBUTION_REWARD}" in sql

    def test_sql_uses_value_updated_at_as_the_time_anchor(self):
        """复用 updated_at 会让 Δt 被无关写入不断归零, 记忆永远衰减不下去。"""
        from app.services.memory.lifecycle import lazy_update

        sql = lazy_update._render_sql("memories_user")
        assert "COALESCE(m.value_updated_at, m.created_at)" in sql
        assert "CURRENT_TIMESTAMP - m.updated_at" not in sql

    def test_sql_matches_singletons_as_category_pairs(self):
        """L1_SINGLETON_SUBS 是 (主类, 子类) 二元组集合, 不是子类名集合。

        当成 text[] 直接传会在驱动层报序列化错, 而调用方吞异常 —— 整个效用值更新
        静默失效, 没有任何日志。这个 bug 单元测试抓不到 (SQL 字符串看着没问题),
        是上线前的 EXPLAIN 验证抓到的。只比子类名也不行: taxonomy 里"其他"这类
        子类在多个主类下都存在。
        """
        from app.services.memory.lifecycle import lazy_update

        sql = lazy_update._render_sql("memories_ai")
        assert "unnest($3::text[], $4::text[]) AS sg(main, sub)" in sql
        assert "sg.main = m.main_category AND sg.sub = m.sub_category" in sql

    def test_singleton_arrays_are_parallel_and_flat(self):
        from app.services.memory.lifecycle.lazy_update import _singleton_arrays
        from app.services.memory.taxonomy import L1_SINGLETON_SUBS

        mains, subs = _singleton_arrays()
        assert len(mains) == len(subs) == len(L1_SINGLETON_SUBS)
        assert all(isinstance(x, str) for x in mains + subs), "元组漏进了数组"
        assert set(zip(mains, subs)) == set(L1_SINGLETON_SUBS)

    def test_sql_breaks_ties_within_a_singleton_group(self):
        """同一条语句里两条同类目记忆可能同时越过 hot 阈值。

        l1_taken 的 EXISTS 看的是语句开始时的表状态, 两条都会读到"还没有 L1",
        于是双双晋升 —— 正好造出要防的第二条 L1。必须再按组内排名去重。
        """
        from app.services.memory.lifecycle import lazy_update

        sql = lazy_update._render_sql("memories_user")
        assert "ROW_NUMBER() OVER" in sql
        assert "PARTITION BY m.user_id, m.workspace_id" in sql
        assert "s.group_rank > 1" in sql

    def test_sql_blocks_a_second_l1_for_singleton_subs(self):
        """惰性更新把层级迁移搬到了热路径, 旧 cron 的 singleton 闸门必须一起搬。

        少了它, 两条"姓名"记忆会同时坐在 L1 上 —— 正是人设分层要消灭的数据损坏。
        """
        from app.services.memory.lifecycle import lazy_update

        sql = lazy_update_sql = lazy_update._render_sql("memories_user")
        assert "l1_taken" in sql
        assert "s.l1_taken OR s.group_rank > 1" in sql
        # 冲突范围必须按 user + workspace 隔离, 否则不同 agent 会互相阻塞晋升
        assert "o.workspace_id IS NOT DISTINCT FROM m.workspace_id" in sql
        assert "o.user_id = m.user_id" in lazy_update_sql


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
        """效用值更新是尽力而为的 —— 它绝不能影响用户拿到回复。"""
        from app.services.memory.lifecycle import lazy_update

        with patch.object(
            lazy_update.db, "execute_raw",
            new=AsyncMock(side_effect=RuntimeError("db down")),
        ):
            assert await lazy_update.record_memory_usage(contributed_ids=["m1"]) == 0

    @pytest.mark.asyncio
    async def test_both_tables_are_targeted(self):
        """记忆 ID 全局唯一, 打两张表比先查归属再定表少一次往返。"""
        from app.services.memory.lifecycle import lazy_update

        with patch.object(
            lazy_update.db, "execute_raw", new=AsyncMock(return_value=1),
        ) as raw:
            await lazy_update.record_memory_usage(contributed_ids=["m1"])
        tables = {
            "memories_user" if "memories_user" in c.args[0] else "memories_ai"
            for c in raw.await_args_list
        }
        assert tables == {"memories_user", "memories_ai"}


def _expected_singleton_args() -> tuple:
    from app.services.memory.lifecycle.lazy_update import _singleton_arrays

    return _singleton_arrays()


class TestSweepIsOnlyABackstop:
    @pytest.mark.asyncio
    async def test_sweep_applies_pure_decay(self):
        """兜底扫描不该给任何回报 —— 它照顾的正是没人用的记忆。"""
        from app.services.memory.lifecycle import lazy_update

        with patch.object(
            lazy_update.db, "execute_raw", new=AsyncMock(return_value=3),
        ) as raw:
            await lazy_update.sweep_stale_values()
        assert raw.await_count == 2, "占位符改写失败会跳过执行"
        sql = raw.await_args_list[0].args[0]
        assert "false AS is_contribution" in sql
        # ID 连接必须被内联查询取代 —— 残留会让扫描按一组不存在的参数执行。
        # 注意 unnest 本身仍在 (singleton 二元组用它), 所以要匹配具体形态。
        assert "unnest($1::text[]) AS id" not in sql
        assert raw.await_args_list[0].args[1:] == _expected_singleton_args()

    @pytest.mark.asyncio
    async def test_sweep_is_bounded(self):
        """无上限的扫描会在大表上长时间持锁。"""
        from app.services.memory.lifecycle import lazy_update

        with patch.object(
            lazy_update.db, "execute_raw", new=AsyncMock(return_value=0),
        ) as raw:
            await lazy_update.sweep_stale_values(limit=100)
        assert "LIMIT 100" in raw.await_args_list[0].args[0]

    @pytest.mark.asyncio
    async def test_sweep_failure_is_reported_but_contained(self):
        from app.services.memory.lifecycle import lazy_update

        with patch.object(
            lazy_update.db, "execute_raw",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            stats = await lazy_update.sweep_stale_values()
        assert stats["scanned"] == 0


def test_half_life_change_requires_rerunning_the_simulation():
    """常数漂了而没重跑推演, 等于闸门结论作废。这里钉住当前值。

    240 是推演选出来的: 180 天在存量人设重排后的最坏场景第 180 天有 -4.2% 退化
    (人设落在 0.72-0.82, 约 140 天就跌破 warm 下行阈值), 240 是消除它的最小值。
    """
    assert HALF_LIFE_DAYS == 240.0
    assert DECAY_LAMBDA == pytest.approx(math.log(2) / 240.0)
