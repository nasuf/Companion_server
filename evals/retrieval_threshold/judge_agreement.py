"""两个评审模型判同一批配对, 量一下标签本身有多可信.

整套 embedding 对比都建立在"这条记忆对回复有没有用"的判定上, 而判定是模型给的,
不是人标的. 如果两个不同的评审模型对同一批配对分歧很大, 那么模型之间几个百分点
的差距就只是在噪声里挑数字, 换谁都一样.

报三个量:
  一致率        两个评审给出同一档的比例
  Cohen's κ    扣掉"瞎猜也会撞上"的部分之后的一致程度. 三档且分布不均时,
                裸一致率会虚高, κ 才是能比的量.
  结论稳健性    换成第二个评审的标签重跑排序对比, 模型排名会不会变.

最后一项最要紧: 即便逐条标签有分歧, 只要两套标签选出同一个赢家, 结论就站得住.

用法:
    python -m evals.retrieval_threshold.judge_agreement --judged /tmp/budget.json \
        --second-judge dashscope:qwen3.5-plus
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path

from evals.retrieval_threshold.judge import JUDGE_PROMPT, parse_verdict
from evals.utility_model.run_eval import build_model

LABELS = ("有用", "沾边", "无用")


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


def _kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[l] / n) * (cb[l] / n) for l in LABELS)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def _top3_capture(pairs: list[dict], key: str, verdict_field: str) -> float:
    by_message: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        by_message[pair["message"]].append(pair)
    by_message = {m: v for m, v in by_message.items() if len(v) >= 4}
    total = hit = 0
    for items in by_message.values():
        total += sum(1 for p in items if p[verdict_field] == "有用")
        hit += sum(1 for p in sorted(items, key=lambda x: -x[key])[:3]
                   if p[verdict_field] == "有用")
    return hit / total if total else 0.0


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", required=True)
    ap.add_argument("--second-judge", default="dashscope:qwen3.5-plus")
    ap.add_argument("--cache", default="/tmp/embed_cache.json")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--json", default="/tmp/agreement.json")
    args = ap.parse_args()

    pairs = [p for p in json.loads(Path(args.judged).read_text()) if p.get("verdict")]
    print(f"{len(pairs)} 条配对, 第二评审 = {args.second_judge}\n")

    model = build_model(args.second_judge)
    sem = asyncio.Semaphore(args.concurrency)
    second = await asyncio.gather(*(_judge(model, p, sem) for p in pairs))
    for pair, verdict in zip(pairs, second):
        pair["verdict2"] = verdict

    graded = [p for p in pairs if p["verdict2"]]
    a = [p["verdict"] for p in graded]
    b = [p["verdict2"] for p in graded]
    agree = sum(1 for x, y in zip(a, b) if x == y) / len(graded)
    print(f"一致率     {agree:.1%}   ({len(graded)} 条双方都给出了判定)")
    print(f"Cohen's κ  {_kappa(a, b):.3f}   "
          f"(<0.2 几乎无一致 / 0.4-0.6 中等 / >0.6 较好)")

    print(f"\n各档分布   {'评审1':>10}{'评审2':>10}")
    ca, cb = Counter(a), Counter(b)
    for label in LABELS:
        print(f"  {label:<8}{ca[label]:>10}{cb[label]:>10}")

    print("\n分歧集中在哪 (评审1 → 评审2)")
    for (x, y), n in Counter(zip(a, b)).most_common():
        if x != y:
            print(f"  {x} → {y}   {n} 条")

    # 结论稳健性: 换一套标签, 模型排名变不变
    cache = json.loads(Path(args.cache).read_text())
    import math

    def cos(u, v):
        d = sum(x * y for x, y in zip(u, v))
        nu = math.sqrt(sum(x * x for x in u))
        nv = math.sqrt(sum(y * y for y in v))
        return d / (nu * nv) if nu and nv else 0.0

    for name, vectors in cache.items():
        for pair in graded:
            pair[name] = cos(vectors[pair["message"]], vectors[pair["memory"]])

    print(f"\n换标签后 top3 排序能力会不会变结论")
    print(f"  {'模型':<26}{'按评审1':>10}{'按评审2':>10}")
    keys = [("bge-m3 (现网)", "sim")] + [(k, k) for k in cache]
    for label, key in keys:
        one = _top3_capture(graded, key, "verdict")
        two = _top3_capture(graded, key, "verdict2")
        print(f"  {label:<26}{one:>10.0%}{two:>10.0%}")

    Path(args.json).write_text(json.dumps(graded, ensure_ascii=False, indent=2))
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    asyncio.run(main())
