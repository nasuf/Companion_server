"""两套记忆生命周期策略的可执行模型, 供离线推演对比.

改分层规则最怕的不是改错公式, 是**改完才发现有用的记忆被衰减掉了** —— 而那要
几个月才看得出来。这里把新旧两套规则都写成纯函数, 在真实记忆快照上向前推演,
让"会不会变差"在动手前就能回答。

两套策略共享一个接口: 给定一条记忆的状态和经过的时间, 返回它的分数与所在层级。
差别集中在三处:

    现行 (l2_dynamics)   分数 = 不可变的初始 importance × 时间档 × 频率档 × 质量档
                         全表夜间重算; 降级需"低于 0.50 持续 30 天"; L3 完全不可检索
    AMV-L 式             V ← min(V·e^(-λΔt) + α·access + β·contrib, Vmax)
                         惰性更新; 上下行分离的滞回阈值; warm 有界采样仍可检索

现行策略的档位是照抄 l2_dynamics 的实现, 不是重新设计 —— 推演和生产口径必须
一致, 否则推演结论不适用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class MemoryState:
    """推演中一条记忆的可变状态。"""

    value: float          # 现行策略下是 current_score, 新策略下是 V(m)
    level: int
    days_since_access: float
    mentions: int
    # 现行策略专用: 连续低于阈值的天数 (对应 Redis 里的 streak)
    low_streak_days: float = 0.0
    # 身份事实 (taxonomy.L1_SINGLETON_SUBS): 姓名/年龄/生日之类。
    # AMV-L 的 hot 层不是永久的 —— 分数跌破下行阈值就降级。对通用记忆这没问题,
    # 对身份事实是错的: 用户一年没问过"你叫什么", 不代表 AI 可以不知道自己叫什么。
    # 所以这类记忆豁免降级, 这是我们相对 AMV-L 的一处刻意偏离。
    protected: bool = False


@dataclass(frozen=True)
class Policy:
    name: str

    def initial(self, importance: float, level: int, days_idle: float,
                protected: bool = False) -> MemoryState:
        raise NotImplementedError

    def step(self, state: MemoryState, days: float, accessed: bool,
             contributed: bool) -> MemoryState:
        raise NotImplementedError

    def is_retrievable(self, state: MemoryState) -> bool:
        raise NotImplementedError


# ── 现行策略 (照抄 l2_dynamics) ────────────────────────────────────────────

def _time_factor(days: float) -> float:
    if days < 30:
        return 1.0
    if days < 90:
        return 0.9
    if days < 180:
        return 0.8
    if days < 365:
        return 0.7
    if days < 730:
        return 0.6
    return 0.5


def _frequency_factor(mentions: int) -> float:
    if mentions <= 2:
        return 1.0
    if mentions <= 5:
        return 1.1
    if mentions <= 10:
        return 1.2
    return 1.3


CURRENT_DEMOTE_BELOW = 0.50
CURRENT_DEMOTE_SUSTAINED_DAYS = 30
CURRENT_PROMOTE_AT = 0.85
CURRENT_PROMOTE_MIN_MENTIONS = 10


@dataclass(frozen=True)
class CurrentPolicy(Policy):
    name: str = "现行 (l2_dynamics)"
    # 升级还要求"用户曾标记重要", 生产上几乎不可能满足 —— 历史 0 次升级。
    # 推演里用它复现现状: False 表示该条件永不满足。
    user_marked_reachable: bool = False

    def initial(self, importance: float, level: int, days_idle: float,
                protected: bool = False) -> MemoryState:
        return MemoryState(
            value=importance, level=level, days_since_access=days_idle, mentions=0,
            protected=protected,
        )

    def step(self, state: MemoryState, days: float, accessed: bool,
             contributed: bool) -> MemoryState:
        idle = 0.0 if accessed else state.days_since_access + days
        mentions = state.mentions + (1 if accessed else 0)
        # 初始 importance 不可变, 每次都从它重算 —— 这是现行实现的关键性质:
        # 分数不累积, 只由"距上次访问多久"和"访问过几次"决定。
        base = state.value if state.level == 1 else state.value
        score = base * _time_factor(idle) * _frequency_factor(mentions)

        level = state.level
        streak = state.low_streak_days
        if level == 2:
            if score < CURRENT_DEMOTE_BELOW:
                streak += days
                if streak >= CURRENT_DEMOTE_SUSTAINED_DAYS:
                    level = 3
            else:
                streak = 0.0
            if (score >= CURRENT_PROMOTE_AT
                    and mentions >= CURRENT_PROMOTE_MIN_MENTIONS
                    and self.user_marked_reachable):
                level = 1
        # L1 永不衰减也永不降级; L3 永不回来 —— 现行实现没有 L3→L2 的路径。
        return replace(state, level=level, days_since_access=idle,
                       mentions=mentions, low_streak_days=streak)

    def is_retrievable(self, state: MemoryState) -> bool:
        return state.level in (1, 2)


# ── AMV-L 式策略 ──────────────────────────────────────────────────────────

# 半衰期 180 天 → λ = ln2/180。选它是为了跟现行策略的衰减速度大致对齐
# (现行: 0.86 的记忆约 730 天跌破 0.50), 这样对比测的是**机制差异**而不是
# "新策略调得更激进所以看起来不同"。
AMVL_HALF_LIFE_DAYS = 180.0
AMVL_LAMBDA = math.log(2) / AMVL_HALF_LIFE_DAYS
AMVL_ACCESS_REWARD = 0.05      # α: 进入候选集
AMVL_CONTRIB_REWARD = 0.12     # β: 真正被注入 prompt (论文要求 β ≥ α)
AMVL_VALUE_MAX = 1.0

# 滞回: 上行阈值高于下行阈值, 防止在边界反复横跳
AMVL_HOT_UP = 0.85
AMVL_HOT_DOWN = 0.78
AMVL_WARM_UP = 0.50
AMVL_WARM_DOWN = 0.42


@dataclass(frozen=True)
class AmvlPolicy(Policy):
    name: str = "AMV-L 式"
    # warm 层的有界采样预算; 0 表示退回"L3 完全不可检索"的现行语义
    warm_sample_budget: int = 3

    def initial(self, importance: float, level: int, days_idle: float,
                protected: bool = False) -> MemoryState:
        # 以初始 importance 为起点, 并把已经闲置的时间一次性折算进去 (惰性衰减:
        # 快照里的记忆早就该衰减了, 只是从没有人算过)
        value = importance * math.exp(-AMVL_LAMBDA * days_idle)
        return MemoryState(
            value=min(value, AMVL_VALUE_MAX), level=level,
            days_since_access=days_idle, mentions=0, protected=protected,
        )

    def step(self, state: MemoryState, days: float, accessed: bool,
             contributed: bool) -> MemoryState:
        value = state.value * math.exp(-AMVL_LAMBDA * days)
        if accessed:
            value += AMVL_ACCESS_REWARD
        if contributed:
            value += AMVL_CONTRIB_REWARD
        value = min(value, AMVL_VALUE_MAX)

        # 滞回: 只有越过"上行阈值"才升, 跌破"下行阈值"才降
        level = state.level
        if state.protected:
            # 身份事实永不降级 —— 见 MemoryState.protected 的说明
            return replace(state, value=value, level=1, days_since_access=0.0
                           if accessed else state.days_since_access + days,
                           mentions=state.mentions + (1 if accessed else 0))
        if level == 1 and value < AMVL_HOT_DOWN:
            level = 2
        elif level == 2:
            if value >= AMVL_HOT_UP:
                level = 1
            elif value < AMVL_WARM_DOWN:
                level = 3
        elif level == 3 and value >= AMVL_WARM_UP:
            # 现行策略没有的路径: 冷记忆被重新用到可以回到 warm
            level = 2

        return replace(
            state, value=value, level=level,
            days_since_access=0.0 if accessed else state.days_since_access + days,
            mentions=state.mentions + (1 if accessed else 0),
        )

    def is_retrievable(self, state: MemoryState) -> bool:
        """只回答"是否在 hot 层"。

        L3 能否进入候选不是这个函数能决定的 —— warm 采样是**有界**的 (κ 条),
        一条 L3 记忆要和同一次查询的其他 L3 候选按相似度竞争那 κ 个名额。这依赖
        查询侧信息, 由推演层的 warm_sample_wins 判定。

        早先这里写成 `level in (1,2) or budget > 0`, 等于宣称预算大于零时全部
        可检索 —— 那让闸门永远返回 100%, 是个不可能失败的测试。
        """
        return state.level in (1, 2)
