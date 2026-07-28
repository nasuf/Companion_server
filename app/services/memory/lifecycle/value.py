"""记忆效用值的惰性更新 (AMV-L §3.2).

## 为什么不再用夜间全表重算

旧实现把分数交给每晚 2:30 的 cron 全表重算。它有两个结构性问题, 都在生产上兑现了:

1. **单点失效且无声**。那个 cron 因为一处 SQL 类型错死了几个月, 期间所有 L2 的
   分数原地冻结、零衰减、零升降级, 没有任何告警。
2. **分数不累积**。每晚都从不可变的初始 importance 重算, 于是"用了一百次"和
   "用了十次"落在同一个频率档里, 使用信号被档位抹平。

惰性更新把衰减挪到**记忆被用到的那一刻**顺带算掉:

    V ← min(V·e^(-λΔt) + α·access + β·contribution, V_max)

Δt 取距 `value_updated_at` 的天数。这样值是累积的, 而且不依赖任何定时任务活着
—— cron 退化为兜底扫描, 只负责照顾那些长期没人碰的记忆。

## 两种使用信号, 权重不同

    access        进了检索候选集 —— 弱证据, 说明它至少跟这轮话题沾边
    contribution  真正被注入 prompt —— 强证据, 它参与了这次回复

AMV-L 要求 β ≥ α。分开记的实际用处在有界采样上: L3 记忆被采样进候选却没能注入
时仍能拿到一点回报, 于是"被反复采到但总差一口气"的记忆会慢慢爬回 warm, 而不是
永远躺在冷层。

## 时间常数怎么定的

半衰期 180 天。这不是拍的 —— 是对齐旧实现的衰减速度: 旧档位下一条 0.86 的记忆
约 730 天跌破 0.50, 新公式在半衰期 180 天时是 0.86·e^(-ln2/180·730) ≈ 0.05,
比旧的快。取 180 天是因为闸门推演显示它在"有用记忆留存"上不劣于旧策略, 同时把
可检索集合压到旧策略的零头。改这个常数必须重跑 evals/memory_lifecycle。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

# ── 衰减与回报 ────────────────────────────────────────────────────────────

HALF_LIFE_DAYS = 180.0
DECAY_LAMBDA = math.log(2) / HALF_LIFE_DAYS

ACCESS_REWARD = 0.05        # α: 进入候选集
CONTRIBUTION_REWARD = 0.12  # β: 真正注入 prompt (AMV-L 要求 β ≥ α)
VALUE_MAX = 1.0

# 两种信号的**数学形式**不同, 不只是权重不同。
#
# 标定时发现纯加法回报有个语义错误: α=0.05 配 180 天半衰期时, 每 30 天进一次候选
# 集的记忆就会一路涨到上限进 hot —— 仅仅"被向量检索捞到过"就足以成为核心记忆。
# 候选集每轮有 50 条, 热门记忆很容易反复入选却从没真正被注入。
#
# 所以 access 改成**趋向天花板的递减回报**: V += α·(ACCESS_CEILING - V)。它能把
# 一条记忆托到天花板附近, 但永远越不过去。天花板取 hot 的下行阈值, 语义正好是
# "被检索到能让你不至于凉掉, 但要成为核心记忆得真的被用上"。
#
# contribution 保持加法, 可以推着记忆穿过 hot 阈值 —— 真正进了 prompt 才算数。
ACCESS_CEILING = 0.78

# ── 滞回阈值 ──────────────────────────────────────────────────────────────
#
# 上行阈值高于下行阈值, 中间是死区。没有死区的话, 一条分数恰在阈值附近的记忆会
# 在"被用到就升、隔天衰减就降"之间反复横跳 —— 每次跳变都要写库、写 changelog,
# 而且用户会感觉 AI 时而记得时而不记得。
#
# 死区宽度 0.07-0.08, 约等于一次 contribution 的回报 (0.12) 的六成: 一次有效使用
# 足以推着记忆穿过死区完成升级, 而单纯的时间波动不足以。
HOT_PROMOTE_AT = 0.85    # L2 → L1
HOT_DEMOTE_AT = 0.78     # L1 → L2
WARM_PROMOTE_AT = 0.50   # L3 → L2
WARM_DEMOTE_AT = 0.42    # L2 → L3


@dataclass(frozen=True)
class ValueUpdate:
    """一次惰性更新的结果。"""

    value: float
    level: int
    changed_level: bool


def decayed_value(value: float, days_elapsed: float) -> float:
    """把闲置时间折算进效用值。"""
    if days_elapsed <= 0:
        return value
    return value * math.exp(-DECAY_LAMBDA * days_elapsed)


def days_since(stamp: datetime | None, fallback: datetime | None = None) -> float:
    """距 stamp 多少天。stamp 为空时退回 fallback, 都为空按 0 处理。

    按 0 处理意味着"当作刚算过", 即不衰减。这是刻意选的保守方向: 时间基准缺失时
    宁可少衰减也不要凭空把记忆打入冷宫。
    """
    anchor = stamp or fallback
    if anchor is None:
        return 0.0
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=UTC)
    return max(0.0, (datetime.now(UTC) - anchor).total_seconds() / 86400)


def next_level(value: float, current_level: int, *, protected: bool = False) -> int:
    """按滞回阈值决定层级。

    protected 用于身份事实 (taxonomy.L1_SINGLETON_SUBS): 姓名/生日这类记忆不该
    因为长期没被问起就掉出 L1 —— 用户一年没问过"你叫什么", 不代表 AI 可以不知道
    自己叫什么。这是相对 AMV-L 的刻意偏离, 论文的 hot 层是纯值驱动的。
    """
    if protected:
        return 1
    if current_level == 1:
        return 2 if value < HOT_DEMOTE_AT else 1
    if current_level == 2:
        if value >= HOT_PROMOTE_AT:
            return 1
        if value < WARM_DEMOTE_AT:
            return 3
        return 2
    # L3: 被重新用到可以回到 warm —— 旧实现没有这条路径, 掉下去就永远回不来。
    return 2 if value >= WARM_PROMOTE_AT else 3


def apply_usage(
    *,
    value: float,
    level: int,
    days_idle: float,
    accessed: bool = False,
    contributed: bool = False,
    protected: bool = False,
) -> ValueUpdate:
    """惰性更新的纯函数核心: 先衰减, 再按使用回报, 最后定层级。

    顺序不能反 —— 先加回报再衰减会把刚拿到的回报也打折, 让高频使用的记忆分数
    偏低。
    """
    updated = decayed_value(value, days_idle)
    if accessed and not contributed:
        # 递减回报, 托到天花板为止 —— 见 ACCESS_CEILING 的说明。
        # 已经高于天花板的记忆不因"又被检索到"而再涨, 但也不会被拉下来。
        updated += ACCESS_REWARD * max(0.0, ACCESS_CEILING - updated)
    if contributed:
        # 注入蕴含"进过候选", 不叠加 —— 同一件事不记两次功。
        updated += CONTRIBUTION_REWARD
    updated = max(0.0, min(VALUE_MAX, updated))

    resolved = next_level(updated, level, protected=protected)
    return ValueUpdate(
        value=updated, level=resolved, changed_level=resolved != level,
    )
