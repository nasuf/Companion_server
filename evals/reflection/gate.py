"""闸门: 反思写进去的判断, 检索有用率不得低于 L2 基线.

反思是唯一往记忆库写**推断**的路径, 而推断错了不报错 —— 它只会安静地待在检索池里,
每次被召回都占掉一个注入名额, 把真正有用的记忆挤出去。所以开启之后要有一个能说
"它在帮忙还是在添乱"的量。

判据沿用已有的检索判定口径 (evals/retrieval_threshold 的 LLM judge, 判"这条记忆
对这句话有没有用"), 这样反思产出和普通记忆比的是同一把尺子:

    L2 基线 (聊天学到的记忆)   有用率 29-37%   ← 2026-07 实测
    建号人设                   有用率 11-20%
    反思判断                   要求 ≥ L2 基线下沿

低于基线说明它在往检索池里灌噪声, 该关掉或改提示词。这跟"洞见读起来有没有道理"
是两回事 —— 一条读着很聪明但从不被用上的判断, 对系统没有价值。

用法:
    # 先导出反思记忆被召回的配对
    python scripts/export_retrieval_pairs.py msgs.json /tmp/pairs.json 1,2
    # 判定后跑闸门
    python -m evals.reflection.gate --pairs /tmp/pairs.json --judged /tmp/judged.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 2026-07 实测的 L2 有用率区间 (n=87, 仅 AI 侧已排除来源混淆)。取下沿做门槛:
# 反思判断至少要和聊天里学到的记忆一样有用, 否则它占的名额是净损失。
L2_BASELINE_LOW = 0.29


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", required=True, help="带 verdict 的检索判定")
    ap.add_argument("--min-samples", type=int, default=20)
    args = ap.parse_args()

    judged = json.loads(Path(args.judged).read_text())
    reflected = [j for j in judged if j.get("provenance") == "reflected"]
    others = [
        j for j in judged
        if j.get("provenance") not in ("reflected", "profile_seed", None)
    ]

    if len(reflected) < args.min_samples:
        print(f"反思记忆只被判定了 {len(reflected)} 条, 不足 {args.min_samples} 条")
        print("样本太少, 结论不可靠 —— 让它多跑几周再评")
        raise SystemExit(0)

    def _rate(rows: list[dict]) -> float:
        useful = sum(1 for r in rows if r.get("verdict") == "有用")
        return useful / len(rows) if rows else 0.0

    reflected_rate = _rate(reflected)
    peer_rate = _rate(others) if others else None

    print(f"反思判断   n={len(reflected):>4}  有用率 {reflected_rate:.1%}")
    if peer_rate is not None:
        print(f"同期其他   n={len(others):>4}  有用率 {peer_rate:.1%}")
    print(f"L2 基线下沿          {L2_BASELINE_LOW:.0%}")

    ok = reflected_rate >= L2_BASELINE_LOW
    print("\n通过 —— 反思产出至少和聊天学到的记忆一样有用" if ok else
          "\n不通过 —— 它在往检索池里灌噪声, 关掉或改提示词")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
