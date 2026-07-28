"""标定惰性衰减的常数 (λ / α / β), 让 hot 集合在真实访问率下有界.

写测试时发现的问题: 半衰期 180 天配 α=0.05 时, 每 6 天进一次候选集的记忆就会
一路涨到上限 —— 回报远大于同期衰减。分数一旦普遍饱和, 阈值就分不出层级, hot 层
会无限膨胀, 而"控制候选集规模"恰恰是分层存在的理由。

这个脚本算的是**平衡值**: 以固定间隔被用到的记忆最终稳定在哪个分数。

    V* = reward / (1 - e^(-λ·interval))

再对照阈值, 看哪些访问频率会落进 hot。目标是: 每周用到几次的进 hot, 几个月才
碰一次的落 warm, 一年不见的沉 cold。
"""

from __future__ import annotations

import math

from app.services.memory.lifecycle.value import (
    ACCESS_REWARD,
    CONTRIBUTION_REWARD,
    HOT_DEMOTE_AT,
    HOT_PROMOTE_AT,
    VALUE_MAX,
    WARM_DEMOTE_AT,
)

INTERVALS_DAYS = (1, 3, 7, 14, 30, 60, 90, 180, 365)


def equilibrium(reward: float, interval_days: float, half_life: float) -> float:
    lam = math.log(2) / half_life
    decay_per_interval = 1 - math.exp(-lam * interval_days)
    if decay_per_interval <= 0:
        return VALUE_MAX
    return min(VALUE_MAX, reward / decay_per_interval)


def tier_of(value: float) -> str:
    if value >= HOT_PROMOTE_AT:
        return "hot"
    if value >= WARM_DEMOTE_AT:
        return "warm"
    return "cold"


def report(half_life: float, alpha: float, beta: float) -> None:
    print(f"\n半衰期 {half_life:.0f} 天  α={alpha}  β={beta}")
    print(f"  {'使用间隔':>10}{'仅进候选':>12}{'被注入':>12}   落层 (按注入)")
    for interval in INTERVALS_DAYS:
        v_access = equilibrium(alpha, interval, half_life)
        v_contrib = equilibrium(beta, interval, half_life)
        print(f"  {interval:>7} 天{v_access:>12.2f}{v_contrib:>12.2f}   "
              f"{tier_of(v_contrib)}")


def main() -> None:
    print("平衡值 = 以该间隔被反复用到的记忆最终稳定在的分数")
    print(f"阈值: hot≥{HOT_PROMOTE_AT} (降级 <{HOT_DEMOTE_AT}), "
          f"cold <{WARM_DEMOTE_AT}")

    print("\n" + "=" * 62)
    print("当前常数")
    print("=" * 62)
    report(180, ACCESS_REWARD, CONTRIBUTION_REWARD)

    print("\n" + "=" * 62)
    print("候选: 缩短半衰期让衰减跟上回报")
    print("=" * 62)
    for half_life in (30, 45, 60, 90):
        report(half_life, ACCESS_REWARD, CONTRIBUTION_REWARD)

    print("\n" + "=" * 62)
    print("候选: 保持半衰期, 调低回报")
    print("=" * 62)
    for alpha, beta in ((0.01, 0.025), (0.02, 0.05)):
        report(180, alpha, beta)


if __name__ == "__main__":
    main()
