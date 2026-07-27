"""收紧注入条数会不会丢记忆 —— 用同一批消息的完整候选列表实测.

抬阈值和收条数是两回事, 混为一谈会得出反向结论:

  抬阈值   按绝对相似度一刀切. 0.52 分的有用记忆直接消失, 哪怕它是这条消息
           最好的匹配. 已实测: 0.50→0.60 每条消息的有用记忆从 1.7 掉到 0.76.
  收条数   按排名保留最好的 N 条. 丢的是尾巴. 只有当有用的没排在前面时才丢.

所以要测的是: **有用的记忆集中在前几名吗?** 之前按相似度分层抽样, 每条消息
只判到 2 条左右, 答不了这个 —— 必须把同一批消息的候选列表整条判完, 才能算出
precision@k 和"保住多少有用记忆".

对照组是"在同等注入量下改用阈值切", 这样两种改法是在同一条成本线上比, 而不是
各自挑一个好看的操作点.

用法:
    python -m evals.retrieval_threshold.budget --pairs /tmp/pairs.json \
        --messages 45 --judge deepseek:deepseek-v4-pro
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import defaultdict
from pathlib import Path

from evals.retrieval_threshold.judge import JUDGE_PROMPT, parse_verdict
from evals.utility_model.run_eval import build_model

PRODUCTION_THRESHOLD = 0.50


async def _judge(model, pair: dict, sem: asyncio.Semaphore) -> str | None:
    owner = "AI 自己" if pair.get("source") == "ai" else "这个用户"
    prompt = JUDGE_PROMPT.format(
        message=pair["message"], memory=pair["memory"], owner=owner
    )
    async with sem:
        for _ in range(2):
            try:
                response = await asyncio.wait_for(model.ainvoke(prompt), timeout=90)
            except Exception:
                continue
            verdict = parse_verdict(getattr(response, "content", "") or str(response))
            if verdict:
                return verdict
    return None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--judge", default="deepseek:deepseek-v4-pro")
    ap.add_argument("--messages", type=int, default=45)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    args = ap.parse_args()

    pairs = json.loads(Path(args.pairs).read_text())
    by_message: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        if pair["sim"] >= PRODUCTION_THRESHOLD:
            by_message[pair["message"]].append(pair)
    for items in by_message.values():
        items.sort(key=lambda p: -p["sim"])

    # 只取候选够多的消息 —— 候选就一两条的消息, 收不收条数都一样, 留在样本里
    # 只会稀释信号.
    eligible = {m: v for m, v in by_message.items() if len(v) >= 4}
    chosen = sorted(eligible)
    if len(chosen) > args.messages:
        chosen = random.Random(args.seed).sample(chosen, args.messages)

    todo = [p for m in chosen for p in eligible[m]]
    print(f"{len(chosen)} 条消息, 候选共 {len(todo)} 条 "
          f"(平均 {len(todo)/len(chosen):.1f} 条/消息), 全部判定中…")

    model = build_model(args.judge)
    sem = asyncio.Semaphore(args.concurrency)
    verdicts = await asyncio.gather(*(_judge(model, p, sem) for p in todo))
    for pair, verdict in zip(todo, verdicts):
        pair["verdict"] = verdict

    ranked = {m: eligible[m] for m in chosen}
    total_useful = sum(
        1 for items in ranked.values() for p in items if p["verdict"] == "有用"
    )
    n_msg = len(ranked)
    print(f"\n候选里共有 {total_useful} 条有用记忆, 分布在 {n_msg} 条消息上\n")

    print("按排名收紧 (保留相似度最高的 N 条)")
    print(f"  {'N':<5}{'保住的有用记忆':>14}{'注入总量':>12}{'精度':>10}{'≥1条有用的消息':>18}")
    for k in (1, 2, 3, 5, 10):
        kept = [p for items in ranked.values() for p in items[:k]]
        useful = sum(1 for p in kept if p["verdict"] == "有用")
        covered = sum(
            1 for items in ranked.values()
            if any(p["verdict"] == "有用" for p in items[:k])
        )
        print(f"  {k:<5}{useful/total_useful:>13.0%}{len(kept)/n_msg:>11.1f}条"
              f"{useful/len(kept):>10.0%}{covered/n_msg:>17.0%}")

    print("\n按阈值收紧 (对照组, 同一批消息)")
    print(f"  {'阈值':<6}{'保住的有用记忆':>13}{'注入总量':>12}{'精度':>10}{'≥1条有用的消息':>18}")
    for cut in (0.50, 0.55, 0.60, 0.65):
        kept = [p for items in ranked.values() for p in items if p["sim"] >= cut]
        if not kept:
            continue
        useful = sum(1 for p in kept if p["verdict"] == "有用")
        covered = sum(
            1 for items in ranked.values()
            if any(p["verdict"] == "有用" and p["sim"] >= cut for p in items)
        )
        print(f"  {cut:<6.2f}{useful/total_useful:>12.0%}{len(kept)/n_msg:>11.1f}条"
              f"{useful/len(kept):>10.0%}{covered/n_msg:>17.0%}")

    print("\n有用记忆落在第几名 (1 = 相似度最高的那条)")
    at_rank: dict[int, int] = defaultdict(int)
    for items in ranked.values():
        for rank, pair in enumerate(items, 1):
            if pair["verdict"] == "有用":
                at_rank[rank] += 1
    running = 0
    for rank in sorted(at_rank):
        running += at_rank[rank]
        print(f"  第{rank:<3}名 {at_rank[rank]:>3} 条   累计 {running/total_useful:.0%}")

    if args.json:
        Path(args.json).write_text(json.dumps(
            [p for items in ranked.values() for p in items],
            ensure_ascii=False, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    asyncio.run(main())
