"""生命周期推演器的性质守卫.

推演器是"改分层规则会不会变差"的唯一判据, 所以它自己出错的代价特别高 —— 一个
永远通过的闸门比没有闸门更危险, 因为它会给出安全的假象。

初版就踩了这个坑: warm 采样被写成 `level in (1,2) or budget > 0`, 于是预算大于
零时所有记忆都"可检索", 闸门恒为 100%。这个文件的第一组测试就是钉住那个 bug。
"""

from __future__ import annotations

import pytest

from evals.memory_lifecycle.policy import (
    AmvlPolicy,
    CurrentPolicy,
    MemoryState,
)


class TestGateCanActuallyFail:
    """闸门必须能失败, 否则它测不出任何东西。"""

    def test_warm_budget_does_not_make_everything_retrievable(self):
        """有界采样是"竞争 κ 个名额", 不是"全部放行"。

        写成后者会让留存率恒为 100%, 闸门永远通过。
        """
        policy = AmvlPolicy(warm_sample_budget=3)
        cold = MemoryState(value=0.1, level=3, days_since_access=400, mentions=0)
        assert not policy.is_retrievable(cold), (
            "L3 被判为可检索 —— 有界采样又被写成无界了"
        )

    def test_pure_decay_degrades_retention(self):
        """不给任何访问信号时, 分数必须真的往下走。否则推演测的是"什么都不变"。"""
        policy = AmvlPolicy()
        state = policy.initial(importance=0.86, level=2, days_idle=0)
        for _ in range(12):
            state = policy.step(state, 30, accessed=False, contributed=False)
        assert state.value < 0.86 * 0.5, f"一年后仍有 {state.value:.3f}, 衰减没生效"
        assert state.level == 3


class TestAmvlValueModel:
    def test_access_resets_the_slide(self):
        """被用到就该回血 —— 这是"使用即价值"的核心假设。"""
        policy = AmvlPolicy()
        idle = policy.initial(0.8, 2, 0)
        used = policy.initial(0.8, 2, 0)
        for _ in range(6):
            idle = policy.step(idle, 30, accessed=False, contributed=False)
            used = policy.step(used, 30, accessed=True, contributed=True)
        assert used.value > idle.value

    def test_contribution_outweighs_mere_access(self):
        """真正进了 prompt 是比"进过候选集"更强的效用证据 (AMV-L 要求 β ≥ α)。"""
        policy = AmvlPolicy()
        base = policy.initial(0.6, 2, 0)
        seen = policy.step(base, 1, accessed=True, contributed=False)
        used = policy.step(base, 1, accessed=True, contributed=True)
        assert used.value > seen.value

    def test_value_is_capped(self):
        """频繁使用不能让分数无限膨胀, 否则它永远降不下来。"""
        policy = AmvlPolicy()
        state = policy.initial(0.9, 1, 0)
        for _ in range(200):
            state = policy.step(state, 1, accessed=True, contributed=True)
        assert state.value <= 1.0

    def test_hysteresis_prevents_oscillation(self):
        """在阈值附近反复横跳会让层级抖动。上行阈值必须高于下行阈值。"""
        from evals.memory_lifecycle.policy import (
            AMVL_HOT_DOWN, AMVL_HOT_UP, AMVL_WARM_DOWN, AMVL_WARM_UP,
        )
        assert AMVL_HOT_UP > AMVL_HOT_DOWN
        assert AMVL_WARM_UP > AMVL_WARM_DOWN

    def test_cold_memories_can_come_back(self):
        """现行策略没有 L3→L2 的路径, 掉下去就永远回不来。新策略必须有。"""
        policy = AmvlPolicy()
        state = MemoryState(value=0.3, level=3, days_since_access=100, mentions=0)
        for _ in range(2):
            state = policy.step(state, 1, accessed=True, contributed=True)
        assert state.level == 2, "冷记忆被反复用到却回不到 warm"

    def test_identity_facts_never_leave_l1(self):
        """AMV-L 的 hot 层会降级, 但身份事实不能 —— 用户一年没问过"你叫什么",
        不代表 AI 可以不知道自己叫什么。这是我们相对 AMV-L 的刻意偏离。"""
        policy = AmvlPolicy()
        state = policy.initial(0.95, 1, days_idle=0, protected=True)
        for _ in range(48):  # 四年不闻不问
            state = policy.step(state, 30, accessed=False, contributed=False)
        assert state.level == 1
        assert state.value < 0.1, "分数该照常衰减, 只是不触发降级"


class TestCurrentPolicyMatchesProduction:
    """推演里的现行策略必须复刻生产实现, 否则对比的是两个都不存在的东西。"""

    def test_l1_never_decays(self):
        policy = CurrentPolicy()
        state = policy.initial(0.95, 1, 0)
        for _ in range(24):
            state = policy.step(state, 30, accessed=False, contributed=False)
        assert state.level == 1

    def test_demotion_needs_a_sustained_streak(self):
        """生产要求"低于 0.50 持续 30 天", 单次低分不该立刻降级。"""
        policy = CurrentPolicy()
        state = MemoryState(value=0.55, level=2, days_since_access=740, mentions=0)
        after_one_day = policy.step(state, 1, accessed=False, contributed=False)
        assert after_one_day.level == 2

    def test_promotion_is_unreachable_as_in_production(self):
        """生产要求"用户曾标记重要", 实测历史 0 次升级。推演要复现这个事实,
        否则会高估现行策略。"""
        policy = CurrentPolicy()
        state = MemoryState(value=0.95, level=2, days_since_access=0, mentions=50)
        for _ in range(12):
            state = policy.step(state, 1, accessed=True, contributed=True)
        assert state.level == 2, "现行策略在推演里升级了, 与生产不符"


def test_factors_match_the_shipped_implementation():
    """档位是照抄 l2_dynamics 的, 不是重新设计 —— 漂移了推演结论就不适用生产。"""
    from app.services.memory.lifecycle import l2_dynamics
    from evals.memory_lifecycle import policy as sim

    for days in (0, 29, 30, 89, 90, 179, 180, 364, 365, 729, 730, 1000):
        assert sim._time_factor(days) == l2_dynamics._time_factor(days), days
    for mentions in (0, 1, 2, 3, 5, 6, 10, 11, 50):
        assert sim._frequency_factor(mentions) == l2_dynamics._frequency_factor(mentions)


@pytest.mark.parametrize("importance,expected_level", [
    (0.95, 1), (0.85, 1), (0.84, 2), (0.50, 2), (0.49, 3),
])
def test_initial_state_respects_the_snapshot_level(importance, expected_level):
    """推演从快照的真实层级出发, 不自己重算 —— 快照里的层级才是生产现状。"""
    policy = AmvlPolicy()
    state = policy.initial(importance, expected_level, days_idle=0)
    assert state.level == expected_level
