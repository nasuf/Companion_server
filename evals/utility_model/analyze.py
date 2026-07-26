"""把两个模型的差距从"点估计"变成"能不能下结论".

三件事:

1. **按用例配对**, 不按调用配对. 每个用例跑 N 次, 这 N 次不是独立样本 —— 同一
   条用例上模型要么会答要么不会答, 把 N 次当独立样本会把置信区间算窄 N 倍.
   独立单位是用例.

2. **聚类 bootstrap**: 对用例(而不是调用)重采样, 给出准确率差值的置信区间.
   区间跨 0 就说明这轮数据不足以判定谁更好, 哪怕点估计有差距.

3. **可疑标注自动识别**: 某条用例上所有模型都跟我的标注不一致时, 更可能是我
   标错了而不是所有模型同时错. 这类用例单独列出来复核, 不然一条错标能让排名
   翻个个儿 (2026-07-25 的「你在干嘛呢」就是这么翻的).
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CaseScore:
    """某模型在某条用例上的表现: N 次里对了几次, 答案稳不稳."""
    correct: int
    total: int
    distinct_answers: int

    @property
    def rate(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def unstable(self) -> bool:
        """同一输入给出过不止一种答案 —— 温度为 0 时仍然发生就是模型侧抖动."""
        return self.distinct_answers > 1


def score_by_case(rows: list[dict[str, Any]]) -> dict[tuple[str, str], CaseScore]:
    """(task, message) → CaseScore. 只统计成功返回的调用."""
    acc: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if "error" in r:
            continue
        acc[(r["task"], r["message"])].append(r)
    out: dict[tuple[str, str], CaseScore] = {}
    for key, items in acc.items():
        out[key] = CaseScore(
            correct=sum(1 for x in items if x["got"] == x["expected"]),
            total=len(items),
            distinct_answers=len({str(x["got"]) for x in items}),
        )
    return out


def suspect_labels(
    by_model: dict[str, dict[tuple[str, str], CaseScore]],
) -> list[tuple[str, str]]:
    """所有模型都在这条用例上全错 —— 先怀疑标注, 再怀疑模型."""
    if len(by_model) < 2:
        return []
    keys = set.intersection(*(set(m) for m in by_model.values()))
    return sorted(
        k for k in keys
        if all(by_model[m][k].correct == 0 for m in by_model)
    )


def paired_compare(
    a: dict[tuple[str, str], CaseScore],
    b: dict[tuple[str, str], CaseScore],
    *,
    exclude: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """A 相对 B 的配对比较 + 聚类 bootstrap 区间."""
    keys = sorted((set(a) & set(b)) - (exclude or set()))
    diffs = [a[k].rate - b[k].rate for k in keys]
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    ties = len(diffs) - wins - losses

    rng = random.Random(20260726)
    boots: list[float] = []
    for _ in range(10000):
        sample = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots))]

    # 单边 bootstrap p: 重采样均值 ≤0 的比例 (A 不优于 B 的经验概率).
    p_not_better = sum(1 for x in boots if x <= 0) / len(boots)

    return {
        "n_cases": len(keys),
        "mean_diff": sum(diffs) / len(diffs) if diffs else 0.0,
        "ci95": (lo, hi),
        "p_not_better": p_not_better,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_cases": [k for k, d in zip(keys, diffs) if d > 0],
        "loss_cases": [k for k, d in zip(keys, diffs) if d < 0],
    }


def stability(scores: dict[tuple[str, str], CaseScore]) -> float:
    """答案稳定的用例占比 —— 判定类调用摇摆本身就是缺陷."""
    if not scores:
        return 0.0
    return sum(1 for s in scores.values() if not s.unstable) / len(scores)
