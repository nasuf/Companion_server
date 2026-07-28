"""闸门 2: L3 有界采样必须提升覆盖而不牺牲精度.

这一步改的是检索语义, 是全流程里唯一会直接改变"哪些记忆进 prompt"的改动, 所以
两个方向都要量:

    精度   进候选集的记忆里, 被判定为"有用"的占比。掉了就是往 prompt 里灌噪声
    覆盖   所有判定为"有用"的记忆里, 能被检索到的占比。这是采样要买的东西

只看精度会得出"什么都别召回最干净"; 只看覆盖会得出"全都召回最全"。有界采样的
意义正在于用可控的精度代价换覆盖, 所以要一起看。

判定数据来自 evals/retrieval_threshold —— 描述的是"这条记忆对那句话有没有用",
与它在哪一层无关, 所以分层改了也能复用。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def evaluate(pairs: list[dict], useful_keys: set[tuple[str, str]],
             hot_threshold: float, warm_threshold: float,
             warm_budget: int) -> dict:
    """按给定策略算精度与覆盖。

    pairs 里每条是 (消息, 记忆, 相似度, 层级)。热层按 hot_threshold 放行;
    冷层要同时满足 warm_threshold 且在该消息的冷层候选里排进前 warm_budget。
    """
    by_message: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        by_message[pair["message"]].append(pair)

    retrieved: set[tuple[str, str]] = set()
    for message, group in by_message.items():
        for pair in group:
            if int(pair.get("level") or 3) in (1, 2):
                if pair["sim"] >= hot_threshold:
                    retrieved.add((message, pair["memory"]))
        if warm_budget > 0:
            cold = sorted(
                (p for p in group if int(p.get("level") or 3) == 3
                 and p["sim"] >= warm_threshold),
                key=lambda p: -p["sim"],
            )
            for pair in cold[:warm_budget]:
                retrieved.add((message, pair["memory"]))

    useful_retrieved = len(retrieved & useful_keys)
    return {
        "retrieved": len(retrieved),
        "precision": useful_retrieved / len(retrieved) if retrieved else 0.0,
        "coverage": useful_retrieved / len(useful_keys) if useful_keys else 0.0,
        "useful_retrieved": useful_retrieved,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="热层配对 (L1/L2)")
    ap.add_argument("--judged", required=True, help="热层判定")
    ap.add_argument("--cold-pairs", help="冷层配对 (L3)")
    ap.add_argument("--cold-judged", help="冷层判定")
    args = ap.parse_args()

    from app.services.memory.retrieval.hybrid import (
        WARM_SAMPLE_BUDGET, WARM_SAMPLE_THRESHOLD, _SIMILARITY_THRESHOLD,
    )

    pairs = json.loads(Path(args.pairs).read_text())
    judged = json.loads(Path(args.judged).read_text())
    if args.cold_pairs:
        pairs = pairs + json.loads(Path(args.cold_pairs).read_text())
    if args.cold_judged:
        judged = judged + json.loads(Path(args.cold_judged).read_text())

    useful_keys = {
        (j["message"], j["memory"]) for j in judged if j.get("verdict") == "有用"
    }
    n_cold = sum(1 for p in pairs if int(p.get("level") or 3) == 3)
    if not n_cold:
        raise SystemExit(
            "配对里没有冷层记忆 —— 这批数据测不了有界采样。老的判定集是从只搜\n"
            "L1+L2 的检索里导出的, 拿它跑只会得到「覆盖 +0%」, 那不是采样无效,\n"
            "是仪器不适用。先用 --cold-pairs / --cold-judged 补上冷层数据。"
        )
    print(f"{len(pairs)} 对配对, 其中冷层 {n_cold} 对; 判定有用 {len(useful_keys)} 对")
    print(f"层级分布: {sorted({int(p.get('level') or 3) for p in pairs})}\n")

    before = evaluate(pairs, useful_keys, _SIMILARITY_THRESHOLD,
                      WARM_SAMPLE_THRESHOLD, warm_budget=0)
    after = evaluate(pairs, useful_keys, _SIMILARITY_THRESHOLD,
                     WARM_SAMPLE_THRESHOLD, warm_budget=WARM_SAMPLE_BUDGET)

    print(f"{'':>14}{'候选数':>9}{'精度':>10}{'覆盖':>10}{'有用命中':>10}")
    for label, row in (("κ=0 (现行)", before), (f"κ={WARM_SAMPLE_BUDGET} (采样)", after)):
        print(f"{label:>14}{row['retrieved']:>9}{row['precision']:>10.1%}"
              f"{row['coverage']:>10.1%}{row['useful_retrieved']:>10}")

    d_prec = after["precision"] - before["precision"]
    d_cov = after["coverage"] - before["coverage"]
    print(f"\n精度 {d_prec:+.1%}   覆盖 {d_cov:+.1%}")

    # 精度容忍 1 个百分点的抖动: 判定集只有几百对, 单条判定的进出就能造成这个量级
    # 的波动。真正的失败信号是精度显著下滑, 或覆盖压根没涨。
    ok_precision = d_prec >= -0.01
    ok_coverage = d_cov > 0
    print("精度: " + ("OK" if ok_precision else "劣化超出容忍"))
    print("覆盖: " + ("OK" if ok_coverage else "没有提升 —— 采样没买到东西"))
    print("\n通过" if ok_precision and ok_coverage else "\n不通过 —— 按计划应退回硬门")


if __name__ == "__main__":
    main()
