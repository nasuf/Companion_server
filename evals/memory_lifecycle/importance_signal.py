"""重测 importance 该不该回到检索排序里.

此前把 importance 从排序公式里摘掉, 是因为**我们的实现坏了**: 75% 的记忆挤在
0.85-0.86 一个窄带里, 一个几乎不变的量乘进排序只会放大噪声。原理本身没问题 ——
Generative Agents 给重要性 2 倍权重。

人设分层 + 摘掉钳位之后 importance 应该重新有真实分布, 所以要重测。这个脚本回答
两件事:

    分布   importance 现在还挤不挤? 挤着就没得谈, 直接维持现状
    信号   把它乘回排序, 判定为有用的记忆能不能排得更靠前?

排序质量用"有用记忆的平均排名"衡量, 并跟两个基线对照: 随机排序 (下界) 和完美排序
(上界)。只报"某公式好了几个点"没有意义 —— 要知道它在可能的区间里挪了多远。
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path


def _rank_quality(groups: list[list[tuple[float, bool]]]) -> float:
    """有用记忆的平均归一化排名, 0 最好 (全排最前), 1 最差。

    按每条消息分组算, 再对消息取平均 —— 不这么做的话, 候选多的消息会主导结果。
    """
    scores: list[float] = []
    for group in groups:
        if not group or not any(useful for _, useful in group):
            continue
        ordered = sorted(group, key=lambda x: -x[0])
        n = len(ordered)
        if n == 1:
            continue
        positions = [i for i, (_, useful) in enumerate(ordered) if useful]
        scores.append(st.mean(positions) / (n - 1))
    return st.mean(scores) if scores else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--judged", required=True)
    args = ap.parse_args()

    pairs = json.loads(Path(args.pairs).read_text())
    judged = json.loads(Path(args.judged).read_text())
    useful = {
        (j["message"], j["memory"]) for j in judged if j.get("verdict") == "有用"
    }

    imps = [float(p.get("importance") or 0) for p in pairs]
    band = sum(1 for i in imps if 0.84 <= i <= 0.87) / len(imps)
    print(f"{len(pairs)} 对配对, importance 中位 {st.median(imps):.3f}, "
          f"标准差 {st.pstdev(imps):.3f}")
    print(f"落在 0.84-0.87 窄带的比例: {band:.0%}")
    if band > 0.5:
        print("→ 仍然挤在一处。一个几乎不变的量乘进排序只会放大噪声, 维持现状。\n")
    else:
        print("→ 分布已经打开, 值得测。\n")

    by_message: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        by_message[pair["message"]].append(pair)

    formulas = {
        "仅相似度 (现行的一部分)": lambda p: p["sim"],
        "相似度 × importance": lambda p: p["sim"] * float(p.get("importance") or 0.5),
        "相似度 × importance²": lambda p: p["sim"] * float(p.get("importance") or 0.5) ** 2,
        "相似度 + 0.2×importance": lambda p: p["sim"] + 0.2 * float(p.get("importance") or 0.5),
    }

    # Phase 1 之后真正携带使用信号的是 current_score, 不是不可变的 importance ——
    # 它才是"这条记忆最近有没有用"的度量。但只有在它真的被填上之后才测得动:
    # 缺值时退回 importance 会算出一个看似在评价 current_score、实则重复了
    # importance 的数字, 那比不测更糟。
    coverage = sum(1 for p in pairs if p.get("current_score") is not None) / len(pairs)
    if coverage >= 0.5:
        formulas["相似度 × current_score"] = (
            lambda p: p["sim"] * float(p["current_score"])
            if p.get("current_score") is not None else p["sim"]
        )
        print(f"current_score 覆盖 {coverage:.0%}, 一并评测\n")
    else:
        print(f"current_score 覆盖仅 {coverage:.0%} —— 惰性衰减刚上线, 还没积累到"
              f"够评测的量。\n跳过它, 等覆盖过半再重跑这个脚本。\n")

    rng = random.Random(0)
    groups_random = [
        [(rng.random(), (m, p["memory"]) in useful) for p in rows]
        for m, rows in by_message.items()
    ]
    groups_perfect = [
        [(1.0 if (m, p["memory"]) in useful else 0.0, (m, p["memory"]) in useful)
         for p in rows]
        for m, rows in by_message.items()
    ]
    lower, upper = _rank_quality(groups_random), _rank_quality(groups_perfect)
    print(f"{'排序公式':<26}{'有用记忆平均排名':>18}{'区间位置':>10}")
    print(f"{'随机 (下界)':<26}{lower:>18.3f}{'0%':>10}")

    for label, fn in formulas.items():
        groups = [
            [(fn(p), (m, p["memory"]) in useful) for p in rows]
            for m, rows in by_message.items()
        ]
        quality = _rank_quality(groups)
        span = (lower - quality) / (lower - upper) if lower > upper else 0.0
        print(f"{label:<26}{quality:>18.3f}{span:>9.0%}")
    print(f"{'完美 (上界)':<26}{upper:>18.3f}{'100%':>10}")

    print("\n判据: 带 importance 的公式若不能明显优于纯相似度, 就不要放回去 ——"
          "\n多一个因子就多一处会漂移、会需要重新标定的东西。")


if __name__ == "__main__":
    main()
