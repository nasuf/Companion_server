"""判定冷层 (L3) 配对的有用性, 供闸门 2 使用.

原有的判定集是从只搜 L1+L2 的检索里导出的, 冷层 0 对 —— 结构上测不了"给 L3 留
κ 个候选名额值不值"。拿它跑闸门 2 只会得到"覆盖 +0%", 那不是采样无效, 是仪器
不适用。

这里用**同一个 judge 提示词**判定冷层配对, 所以两批结果可以直接比。判定的是
"这条记忆对这句话有没有用", 与它在哪一层无关。

只判定过门的那些配对: 门下的本来就进不了候选集, 判它们不影响闸门结论, 徒增开销。

用法:
    python -m evals.memory_lifecycle.judge_cold_pairs \\
        --pairs /tmp/cold_pairs.json --out /tmp/cold_judged.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from evals.retrieval_threshold.judge import JUDGE_PROMPT, parse_verdict
from evals.utility_model.run_eval import build_model


async def _judge_one(model, pair: dict, sem: asyncio.Semaphore) -> str | None:
    owner = "AI 自己" if pair.get("source") == "ai" else "这个用户"
    prompt = JUDGE_PROMPT.format(
        message=pair["message"], memory=pair["memory"], owner=owner,
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge", default="deepseek:deepseek-v4-pro")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--threshold", type=float, default=None,
                    help="缺省取生产的冷层门")
    args = ap.parse_args()

    if args.threshold is None:
        from app.services.memory.retrieval.hybrid import WARM_SAMPLE_THRESHOLD

        args.threshold = WARM_SAMPLE_THRESHOLD

    pairs = json.loads(Path(args.pairs).read_text())
    eligible = [p for p in pairs if p["sim"] >= args.threshold]
    print(f"{len(pairs)} 对冷层配对, 过 {args.threshold} 门的 {len(eligible)} 对待判定")

    model = build_model(args.judge)
    sem = asyncio.Semaphore(args.concurrency)
    verdicts = await asyncio.gather(
        *[_judge_one(model, p, sem) for p in eligible]
    )

    judged = [
        {**pair, "verdict": verdict}
        for pair, verdict in zip(eligible, verdicts) if verdict
    ]
    Path(args.out).write_text(json.dumps(judged, ensure_ascii=False, indent=2))

    counts = Counter(j["verdict"] for j in judged)
    print(f"判定成功 {len(judged)}/{len(eligible)}: {dict(counts)}")
    if judged:
        useful = counts.get("有用", 0)
        print(f"冷层过门配对的有用率 {useful / len(judged):.1%}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
