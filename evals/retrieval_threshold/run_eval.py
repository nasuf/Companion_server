"""标定记忆检索的相似度阈值.

生产阈值 `_SIMILARITY_THRESHOLD = 0.50` 是从 spec 的 0.7 下调来的, 理由是 bge-m3
对中文短文本召回弱. 真实流量上的观察是 97% 的消息都能在 0.50 上匹配到东西
(包括 "Hello, hello" 命中 "我是汉族"), 说明下调的代价落在精度上, 而这个代价
一直没被量过.

做法: 对真实配对按相似度分层抽样, 逐条评审"这条记忆对回复有没有用", 然后算
每个候选阈值下的精度与覆盖:

    精度 = 放行的配对里有用的占比        —— 注入 prompt 的记忆有多少是真能用的
    覆盖 = 至少还留下一条有用记忆的消息   —— 抬阈值会丢掉多少本该召回的轮次

两者对着看才能选阈值: 只看精度会一路抬到没东西可召, 只看覆盖就是现状.

用法:
    python -m evals.retrieval_threshold.run_eval --pairs /tmp/pairs.json \
        --judge deepseek:deepseek-v4-pro --per-band 45
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

BANDS = [(0.45, 0.50), (0.50, 0.55), (0.55, 0.60),
         (0.60, 0.65), (0.65, 0.70), (0.70, 2.0)]
CANDIDATE_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70)


def _band_of(sim: float) -> tuple[float, float] | None:
    for lo, hi in BANDS:
        if lo <= sim < hi:
            return (lo, hi)
    return None


async def _judge_one(model, pair: dict, sem: asyncio.Semaphore) -> str | None:
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
    ap.add_argument("--per-band", type=int, default=45)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    args = ap.parse_args()

    pairs = json.loads(Path(args.pairs).read_text())
    by_band: dict[tuple[float, float], list[dict]] = defaultdict(list)
    for pair in pairs:
        band = _band_of(pair["sim"])
        if band:
            by_band[band].append(pair)

    rng = random.Random(args.seed)
    sample: list[dict] = []
    for band, items in by_band.items():
        picked = items if len(items) <= args.per_band else rng.sample(items, args.per_band)
        for p in picked:
            sample.append({**p, "band": band})
    print(f"抽样 {len(sample)} 条配对, 覆盖 {len(by_band)} 个相似度带")

    model = build_model(args.judge)
    sem = asyncio.Semaphore(args.concurrency)
    verdicts = await asyncio.gather(*(_judge_one(model, p, sem) for p in sample))
    for pair, verdict in zip(sample, verdicts):
        pair["verdict"] = verdict

    graded = [p for p in sample if p["verdict"]]
    print(f"评审成功 {len(graded)}/{len(sample)}\n")

    print(f"{'相似度带':<14}{'n':>5}{'有用':>8}{'沾边':>8}{'无用':>8}")
    for band in BANDS:
        rows = [p for p in graded if p["band"] == band]
        if not rows:
            continue
        n = len(rows)
        counts = {v: sum(1 for r in rows if r["verdict"] == v) for v in ("有用", "沾边", "无用")}
        label = f"{band[0]:.2f}-{band[1]:.2f}" if band[1] < 2 else f"{band[0]:.2f}+"
        print(f"{label:<14}{n:>5}"
              + "".join(f"{counts[v]/n:>7.0%} " for v in ("有用", "沾边", "无用")))

    # 各带抽样率不同 (高相似度带样本少, 被全取), 直接对样本求平均会高估精度.
    # 按该带在全量配对里的真实数量加权还原.
    useful_rate: dict[tuple[float, float], float] = {}
    population: dict[tuple[float, float], int] = {}
    for band in BANDS:
        rows = [p for p in graded if p["band"] == band]
        population[band] = len(by_band.get(band, []))
        useful_rate[band] = (
            sum(1 for r in rows if r["verdict"] == "有用") / len(rows) if rows else 0.0
        )

    verdict_of = {(p["message"], p["memory"]): p["verdict"] for p in graded}
    messages = {p["message"] for p in pairs}
    print(f"\n{'阈值':<8}{'放行精度(按量加权)':>20}{'仍有可用记忆的消息':>22}{'平均注入条数':>14}")
    for cut in CANDIDATE_THRESHOLDS:
        bands_kept = [b for b in BANDS if b[0] >= cut]
        total = sum(population[b] for b in bands_kept)
        useful = sum(population[b] * useful_rate[b] for b in bands_kept)
        precision = useful / total if total else 0.0
        covered = sum(
            1 for m in messages
            if any(verdict_of.get((m, p["memory"])) == "有用"
                   for p in pairs if p["message"] == m and p["sim"] >= cut)
        )
        print(f"{cut:<8.2f}{precision:>19.0%}{covered/len(messages):>21.0%}"
              f"{total/len(messages):>14.1f}")

    print("\n注: 覆盖只统计抽样中判为有用的配对, 是下界 —— 未被抽到的配对里还有\n"
          "    有用的, 所以真实覆盖高于表中数值, 但各阈值间的相对关系不受影响.")

    if args.json:
        Path(args.json).write_text(json.dumps(graded, ensure_ascii=False, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    asyncio.run(main())
